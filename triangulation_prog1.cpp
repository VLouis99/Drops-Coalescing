#include <iostream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <cmath>
#include <fstream>
#include <iterator>
#include <vector>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Triangulation;
typedef K::Point_2 Point;
typedef Triangulation::Vertex_handle Vertex_handle;
typedef Triangulation::Vertex_circulator Vertex_circulator;

// Global Variables
int n = 10000; // Number Of Drops
float theta = 90 ; // Angle of Plaque
float init_r = 1e-5 ; // Initial Radius
int iterations =  int(n-1); // How many coalescence events

// Initialisation of vectors that will contain the scalars
std :: vector<float> times ;
std :: vector<float> radii ;
std :: vector<float> means;
std :: vector<float> mins;
std :: vector<float> maxs;
std :: vector<float> vertices_n ;
std :: vector <float> surfacecovered ;
std::vector<std::vector<float>> distribution;

// Structure of a Point/Drop
struct PointWithNeighbors {
    Point p; // Coordinates
    double r; // Radius of Drop
    std::vector<PointWithNeighbors*> neighbors; // Neighbours of said drop
    Point coal_p ; // Coordinates of the nearest drop that it will coalesce with if it still exists
    double t ; // Coalescence Time
    float probablity ; // Probability of jumping ( Not finished )

};

// Function to implement chance that the drop will jump (Unused)
double calc_prob(double radius,double R_min) {
    double gamma = 1 ;
    double p = (pow(radius, gamma))/(pow(R_min,gamma)) ;
    return p ;
}
// Function to calculate the mass of a drop
float mass(PointWithNeighbors p1) {
    float theta_rad = theta *     M_PI / 180;
    float ftheta = (2-3 * cos(theta_rad) + pow(cos(theta_rad),3.0)) / (3* pow(sin(theta_rad),3.0));
    float r = p1.r;
    return M_PI * ftheta * pow(r,3.0) ;
}

// Function that calculates the center of mass between two points/drops
Point center_of_mass(PointWithNeighbors p1, PointWithNeighbors p2) {
    float m1 = mass(p1);
    float m2 = mass(p2);
    float xc = (m1 * p1.p.x() + m2 * p2.p.x()) / (m1 + m2);
    float yc = (m1 * p1.p.y() + m2 * p2.p.y()) / (m1 + m2);
    return Point(xc,yc);

}

//Function that returns neighbours of a point using CGAL circulators
std::vector<Vertex_handle> get_neighbours(Triangulation &T,Vertex_handle &v) {

    Vertex_circulator vcirc = T.incident_vertices(v);
    std::vector <Vertex_handle> neighbors;
    do {
        neighbors.push_back(vcirc);
    } while (++vcirc != T.incident_vertices(v));
    return neighbors ;
}

// Delaunay Triangulation T
template<typename T>

//Function that calculates the average of a vector (List)
double getAverage(std::vector<T> const& v) {
    if (v.empty()) {
        return 0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Function that changes/updates the neighbours of a given singular point/drop and also finds the closest neighbour
void change_neighbours(Triangulation& T,std::vector<PointWithNeighbors>& points_with_neighbors,PointWithNeighbors &pw){
    auto vertex_iter = T.nearest_vertex(pw.p) ; // Finds the Vertex Handle associated to the concerned drop
    pw.neighbors.clear() ; // Clearing all old neighbours
    auto neighbor_circulator = T.incident_vertices(vertex_iter); // CGAL Circulator
    do {
        if (T.is_infinite(neighbor_circulator)) {
            continue;
        }

        // Find the PointWithNeighbors object corresponding to the neighbor vertex
        auto neighbor_vertex = neighbor_circulator->handle();
        auto neighbor_pw_iter = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                             [&neighbor_vertex](const PointWithNeighbors& pw) { return pw.p == neighbor_vertex->point(); });
        if (neighbor_pw_iter != points_with_neighbors.end()){
            pw.neighbors.push_back(&(*neighbor_pw_iter));
        }
    } while (++neighbor_circulator != T.incident_vertices(vertex_iter));

    // Calculate the closest neighbor and the corresponding coal point and time
    double min_dist = std::numeric_limits<double>::max();
    for (const auto &neighbor_pw: pw.neighbors) {
        double dist = pow( sqrt(pow(pw.p.x()- neighbor_pw->p.x(),2) + pow(pw.p.y()- neighbor_pw->p.y(),2) ) / (neighbor_pw->r + pw.r) ,3 ) * times.back() ;
        if (dist < min_dist && dist > 0.0000000000000001) {
            min_dist = dist;
            pw.coal_p = neighbor_pw->p;
            pw.t = pow( sqrt(pow(pw.p.x()- neighbor_pw->p.x(),2) + pow(pw.p.y()- neighbor_pw->p.y(),2) ) / (neighbor_pw->r + pw.r) ,3 ) * times.back() ;
        }
        if (dist < neighbor_pw->r + pw.r){
            min_dist = 0 ;
            pw.coal_p = neighbor_pw->p;
            pw.t = times.back() ;
        }
    }
}


// Function that changes/updates all the neighbours of every single drop/point
void update_all_neighbors(Triangulation& T, std::vector<PointWithNeighbors>& points_with_neighbors) {
    for (auto& neigh : points_with_neighbors) {
        change_neighbours(T, points_with_neighbors, neigh);
    }
}

// Function that changes/coalesceces two drops which are 'touching' or overlapping
void touching_points(Triangulation& T, std::vector<PointWithNeighbors>& points_with_neighbors,PointWithNeighbors& pw1, PointWithNeighbors& pw2, int &i) {
    if (i <= 1) { return; } // Checks if there is a neighbour
    Vertex_handle v, v1;

    // Checks whether we have two different points and if yes, creates vertex handles
    if (pw1.p != pw2.p) {
        v = T.nearest_vertex(pw1.p);
        v1 = T.nearest_vertex(pw2.p);
    }
    else {
        return;
    }
    // Checks if the two vertex handles are different ( May happen when both vertices are inifinite )
    if (v == v1) {
        std::cout << "The vertices are the same." << std::endl;
        i = i+1;
        return ;
    }
    i = i-1 ;
    if (i<=1) { return ;}

    times.push_back(times.back()) ; // The time will be the same as the last element (in the list) because we consider it instantaneous if two drops are overlapping
    Point new_point = center_of_mass(pw1,pw2) ;

    // Insert the new point into the Delaunay triangulation and the points_with_neighbors vector
    PointWithNeighbors pw;
    pw.p = new_point;
    pw.r = pow(pow(pw2.r,3.0) + pow(pw1.r,3.0),(1.0/3.0));

    //Save the neigbours of the two points to be removed
    auto neigh_removed_point = get_neighbours(T,v) ;
    auto neigh_removed_point2 = get_neighbours(T,v1) ;

    if (neigh_removed_point == neigh_removed_point2) {
        std::cout << "The neighbours are the same." << std::endl;
        return ;
    }

    // Remove from triangulation

    T.remove(v);
    T.remove(v1) ;
    T.insert(new_point) ;


    // Remove the two points from the points_with_neighbors vector
    auto x_coal_remove1 = pw1.p.x() ;
    auto it_other1 = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                  [x_coal_remove1](PointWithNeighbors& point) {return point.p.x() == x_coal_remove1; });
    points_with_neighbors.erase(it_other1) ;

    auto x_coal_remove = pw2.p.x() ;
    auto it_other = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                 [x_coal_remove](PointWithNeighbors& point) {return point.p.x() == x_coal_remove; });
    points_with_neighbors.erase(it_other) ;

    // Add the new point in the vector
    points_with_neighbors.push_back(pw);
    change_neighbours(T,points_with_neighbors,pw) ;
    //assert(pw.neighbors != NULL);

    // Calculate the closest neighbor and the corresponding coal point and time
    double min_dist = std::numeric_limits<double>::max();
    for (const auto &neighbor_pw: pw.neighbors) {
        double dist = pow( sqrt(pow(pw.p.x()- neighbor_pw->p.x(),2) + pow(pw.p.y()- neighbor_pw->p.y(),2) ) / (neighbor_pw->r + pw.r) ,3 ) * times.back() ;
        if (dist < min_dist && dist > 0.0000000000000001) {
            min_dist = dist;
            pw.coal_p = neighbor_pw->p;
            pw.t = pow( sqrt(pow(pw.p.x()- neighbor_pw->p.x(),2) + pow(pw.p.y()- neighbor_pw->p.y(),2) ) / (neighbor_pw->r + pw.r) ,3 ) * times.back() ;
        }
    }

    // Change all the neighbours in the vector list who were connected to either coal_remove or removed_point
    for (auto v : neigh_removed_point){
        for( auto &e : points_with_neighbors) {
            if (e.p == v->point()) {change_neighbours(T,points_with_neighbors,e) ;}
        }

    }

    for (auto v : neigh_removed_point2){
        for( auto &e : points_with_neighbors) {
            if (e.p == v->point()) {change_neighbours(T,points_with_neighbors,e)  ;}
        }
    }
    radii.clear() ; // Clearing the radii list which contains all the Radiusus of the points/drops

// Grow all the Drops and add all the radiusus of all points into the raddii vector
    for (PointWithNeighbors& pt : points_with_neighbors) {
        pt.r = pow(times.end()[-1]/times.end()[-2],1.0/3.0) * pt.r;
        radii.push_back(pt.r) ;
    }
    // Check whether or not the newly formed drop is overlapping any other drops and if so, calls touching_points
    for (const auto &neighbor_pw2: pw.neighbors) {
        if (pow(sqrt(pow(pw.p.x() - neighbor_pw2->p.x(), 2) + pow(pw.p.y() - neighbor_pw2->p.y(), 2)), 2) <
            (neighbor_pw2->r + pw.r)) {
            auto x_coal_remove2 = neighbor_pw2->p.x(); // getting the x coordinates of the neighbour who is too close
            auto it_other2 = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                          [x_coal_remove2](PointWithNeighbors &point) {
                                              return point.p.x() == x_coal_remove2;
                                          }); // Getting its iterator
            if (it_other2 != points_with_neighbors.end()) {
                touching_points(T, points_with_neighbors, pw, *it_other2, i);
                break ;
            }
        }
    }

    for (PointWithNeighbors& pt : points_with_neighbors) {
        /*
        pt.r = pow(times.end()[-1]/times.end()[-2],1.0/3.0) * pt.r;
        */
        radii.push_back(pt.r) ;
    }

    // Calculating the statistics of the drops (Max size,surface,minimum size etc)
    double max = *max_element(radii.begin(), radii.end());
    double min= *min_element(radii.begin(), radii.end());
    mins.push_back(min);
    maxs.push_back(max);
    float power_three = 0 ;
    float power_two = 0 ;
    float surface = 0 ;
    for (auto i : radii){
        power_three += pow(i,3.0) ;
        power_two += pow(i,2.0) ;
        surface += M_PI * pow(i,2.0) ;
    }
    surfacecovered.push_back(surface);
    means.push_back(float(power_three/power_two)) ;
    vertices_n.push_back(T.number_of_vertices()) ;
    radii.clear() ;


    // Write values to separate text files
    /*
    std::ofstream timesFile("times.txt", std::ios::app);
    if (timesFile.is_open()) {
        timesFile << (times.back()) << std::endl;
        timesFile.close();
    } else {
        std::cout << "Unable to open times.txt file." << std::endl;
    }

    std::ofstream surfaceFile("surface.txt", std::ios::app);
    if (surfaceFile.is_open()) {
        surfaceFile << surface << std::endl;
        surfaceFile.close();
    } else {
        std::cout << "Unable to open surface.txt file." << std::endl;
    }

    std::ofstream meansFile("means.txt", std::ios::app);
    if (meansFile.is_open()) {
        meansFile << float(power_three / power_two) << std::endl;
        meansFile.close();
    } else {
        std::cout << "Unable to open means.txt file." << std::endl;
    }

    std::ofstream maxFile("max.txt", std::ios::app);
    if (maxFile.is_open()) {
        maxFile<< max << std::endl;
        maxFile.close();
    } else {
        std::cout << "Unable to open max.txt file." << std::endl;
    }

    std::ofstream minsFile("min.txt", std::ios::app);
    if (minsFile.is_open()) {
        minsFile << min << std::endl;
        minsFile.close();
    } else {
        std::cout << "Unable to open min.txt file." << std::endl;
    }
    */



}


// A function to remove the point with the smallest t value and add a new point (Similar to touching_points function above)
void remove_point_with_min_t_and_add_new_point(Triangulation& T, std::vector<PointWithNeighbors>& points_with_neighbors, int &i) { // const Point& new_point
    i = i - 1;
    if (i <= 1) { return; }
    // Find the PointWithNeighbors object with the smallest time t
    auto it_min_t = std::min_element(points_with_neighbors.begin(), points_with_neighbors.end(),
                                     [](const PointWithNeighbors &a, const PointWithNeighbors &b) {
                                         return a.t < b.t;
                                     });

    PointWithNeighbors removed_point = *it_min_t; // Point struct
    times.push_back(removed_point.t + times.back());






    PointWithNeighbors coal_remove;
    // Get the structure of the coal_p of the point removed
    for (auto &e: points_with_neighbors) {
        if (e.p == removed_point.coal_p) {
            coal_remove = e;
            break;
        }
    }


    Point new_point = center_of_mass(*it_min_t, coal_remove);


    // Grow all the Drops
    for (PointWithNeighbors& pt : points_with_neighbors) {
        pt.r = pow(times.end()[-1]/times.end()[-2],1.0/3.0) * pt.r;
        radii.push_back(pt.r) ;
    }


    // Insert the new point into the Delaunay triangulation and the points_with_neighbors vector
    PointWithNeighbors pw;
    pw.p = new_point;
    pw.r = pow(pow(coal_remove.r, 3.0) + pow(removed_point.r, 3.0), (1.0 / 3.0));

    // Remove the closest points from the Delaunay triangulation
    Vertex_handle v = T.nearest_vertex(removed_point.p);
    Vertex_handle v1 = T.nearest_vertex(coal_remove.p);


    //Save the neigbours of the two points to be removed
    auto neigh_removed_point = get_neighbours(T, v);
    auto neigh_removed_point2 = get_neighbours(T, v1);
    // Remove from triangulation

    T.remove(v);
    T.remove(v1);
    T.insert(new_point);


    // Remove the closest point from the points_with_neighbors vector
    points_with_neighbors.erase(it_min_t);

    auto x_coal_remove = coal_remove.p.x();
    auto it_other = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                 [x_coal_remove](PointWithNeighbors &point) { return point.p.x() == x_coal_remove; });

    //assert(*it_min_t != *it_other) ;
    points_with_neighbors.erase(it_other);


    // Add the point in the vector
    points_with_neighbors.push_back(pw);


    // Change all the neighbours in the vector list who were connected to either coal_remove or removed_point and change neighbours of the new added point
    change_neighbours(T, points_with_neighbors, pw);
    for (auto v: neigh_removed_point) {
        for (auto &e: points_with_neighbors) {
            if (e.p == v->point()) { change_neighbours(T, points_with_neighbors, e); }
        }

    }

    for (auto v: neigh_removed_point2) {
        for (auto &e: points_with_neighbors) {
            if (e.p == v->point()) { change_neighbours(T, points_with_neighbors, e); }
        }
    }

    // Calculate the closest neighbor and the corresponding coal point and time
    double min_dist = std::numeric_limits<double>::max();
    for (const auto &neighbor_pw: pw.neighbors) {
        double dist = pow(sqrt(pow(pw.p.x() - neighbor_pw->p.x(), 2) + pow(pw.p.y() - neighbor_pw->p.y(), 2)) /
                          (neighbor_pw->r + pw.r), 3) * times.back();
        if (dist < min_dist && dist > 0.0000000000000001) {
            min_dist = dist;
            pw.coal_p = neighbor_pw->p;
            pw.t = pow(sqrt(pow(pw.p.x() - neighbor_pw->p.x(), 2) + pow(pw.p.y() - neighbor_pw->p.y(), 2)) /
                       (neighbor_pw->r + pw.r), 3) * times.back();
        }
    }
    radii.clear() ;

// Grow all the Drops

    for (PointWithNeighbors& pt : points_with_neighbors) {
        /*
        pt.r = pow(times.end()[-1]/times.end()[-2],1.0/3.0) * pt.r;
         */
        radii.push_back(pt.r) ;
    }

    double max = *max_element(radii.begin(), radii.end());
    double min= *min_element(radii.begin(), radii.end());

    mins.push_back(min);
    maxs.push_back(max);
    float power_three = 0 ;
    float power_two = 0 ;
    float surface = 0 ;
    for (auto i : radii){
        power_three += pow(i,3.0) ;
        power_two += pow(i,2.0) ;
        surface += M_PI * pow(i,2.0) ;
    }
    surfacecovered.push_back(surface);
    means.push_back(float(power_three/power_two)) ;
    vertices_n.push_back(T.number_of_vertices()) ;

    /*
    // Write values to separate text files
    std::ofstream timesFile("times.txt", std::ios::app);
    if (timesFile.is_open()) {
        timesFile << (removed_point.t + times.back()) << std::endl;
        timesFile.close();s
    } else {
        std::cout << "Unable to open times.txt file." << std::endl;
    }

    std::ofstream surfaceFile("surface.txt", std::ios::app);
    if (surfaceFile.is_open()) {
        surfaceFile << surface << std::endl;
        surfaceFile.close();
    } else {
        std::cout << "Unable to open surface.txt file." << std::endl;
    }

    std::ofstream meansFile("means.txt", std::ios::app);
    if (meansFile.is_open()) {
        meansFile << float(power_three / power_two) << std::endl;
        meansFile.close();
    } else {
        std::cout << "Unable to open means.txt file." << std::endl;
    }

    std::ofstream maxFile("max.txt", std::ios::app);
    if (maxFile.is_open()) {
        maxFile<< max << std::endl;
        maxFile.close();
    } else {
        std::cout << "Unable to open max.txt file." << std::endl;
    }

    std::ofstream minsFile("min.txt", std::ios::app);
    if (minsFile.is_open()) {
        minsFile << min << std::endl;
        minsFile.close();
    } else {
        std::cout << "Unable to open min.txt file." << std::endl;
    }
     */
    radii.clear() ;


    for (const auto &neighbor_pw2: pw.neighbors) {
        if (pow(sqrt(pow(pw.p.x() - neighbor_pw2->p.x(), 2) + pow(pw.p.y() - neighbor_pw2->p.y(), 2)), 2) <
            (neighbor_pw2->r + pw.r)) {
            auto x_coal_remove2 = neighbor_pw2->p.x(); // getting the x coordinates of the neighbour who is too close
            auto it_other2 = std::find_if(points_with_neighbors.begin(), points_with_neighbors.end(),
                                          [x_coal_remove2](PointWithNeighbors &point) {
                                              return point.p.x() == x_coal_remove2;
                                          }); // Getting its iterator
            if (it_other2 != points_with_neighbors.end()) {
                touching_points(T, points_with_neighbors, pw, *it_other2, i);
                break ;
            } else {
                std::cout << 'It was the last element' << std::endl;
            }
        }
    }


}



// Main Code
int main()
{
    std::vector<Point> points;
    CGAL::Random_points_in_square_2<Point> gen(1.0); // Square of size 1x1
    times.push_back(1) ; // Adding the initial time which is set to 1 here

    //Adding all the  Coordinates to a vector
    for(int i=0; i<n; i++)
    {
        points.push_back(*gen++);
    }

    // Calculate the Delaunay triangulation
    Triangulation T;
    T.insert(points.begin(), points.end()); // Inserting all the points into the Trianguation T

    // Output some information about the triangulation
    std::cout << "Number of vertices: " << T.number_of_vertices() << std::endl;
    std::cout << "Number of faces: " << T.number_of_faces() << std::endl;



    // INITIALISING THE STRUCTURE
    std::vector<PointWithNeighbors> points_with_neighbors;
    for (auto vertex_iter = T.finite_vertices_begin(); vertex_iter != T.finite_vertices_end(); ++vertex_iter) {
        // Create a PointWithNeighbors object for the current vertex
        PointWithNeighbors pw;
        pw.p = vertex_iter->point();
        pw.r = init_r;
        // Add the fully initialized PointWithNeighbors object to the vector
        points_with_neighbors.push_back(pw);
    }


    // Adding the vector list of neighbours to each point
    for (auto &pw : points_with_neighbors){
        // Iterate over the neighbors of the current vertex
        change_neighbours(T,points_with_neighbors,pw) ;
    }
    // Number of iterations ( This is to put a limit incase, because if the number above is used, touching points does not necessarily take into account the recursivity of the touching points function )
    int i  = iterations ;

    // Create a map to store distributions for each desired time
    std::map<double, std::vector<int>> timeToHistogram;

    do {
        remove_point_with_min_t_and_add_new_point(T,points_with_neighbors,i);
        /*
        for (const auto &point_with_neighbors : points_with_neighbors) {
            if (point_with_neighbors.neighbors.empty()) {
                std::cout << "Point at (" << point_with_neighbors.p.x() << ", " << point_with_neighbors.p.y() << ") has no neighbors." << std::endl;
            }
        }
        */
        /*
        // Calculating the Distributions
        const int num_bins = 50; // Number of bins in the histogram
        if (i % 1000 == 0) {
            // Create a histogram for the current time 't'
            std::vector<int> radiusHistogram(num_bins, 0);
            double max_radius = 0.0; // Initialize to a suitable value

            // Calculate the maximum radius
            for (const auto &pw : points_with_neighbors) {
                if (pw.t == i) {
                    max_radius = std::max(max_radius, pw.r);
                }
            }

            // Calculate the bin width
            double bin_width = max_radius / num_bins;

            // Fill the histogram with radii
            for (const auto &pw : points_with_neighbors) {
                if (pw.t == i) {
                    int bin_index = static_cast<int>(std::ceil(pw.r / bin_width));
                    if (bin_index < num_bins) {
                        radiusHistogram[bin_index]++;
                    }
                }
            }

            // Store the histogram for the current time 't'
            timeToHistogram[i] = radiusHistogram;

            // Save the histogram to a text file
            std::ofstream hist_file("histogram_" + std::to_string(i) + ".txt");
            for (int bin_index = 0; bin_index < num_bins; bin_index++) {
                hist_file << bin_index * bin_width << " " << radiusHistogram[bin_index] << "\n";
            }

        }*/


    } while( i >= 5);

    // Saving the statistics into different text files
    std::cout << "Number of vertices: " << T.number_of_vertices() << std::endl;
    std::cout << "Number of faces: " << T.number_of_faces() << std::endl;

    std::ofstream output_file("/home/anamaria/Documents/timee.txt");
    std::ostream_iterator<float>output_iterator(output_file,"\n");
    std::copy(std::begin(times),std::end(times),output_iterator);

    std::ofstream output_file1("/home/anamaria/Documents/meane.txt");
    std::ostream_iterator<float>output_iterator1(output_file1,"\n");
    std::copy(std::begin(means),std::end(means),output_iterator1);

    std::ofstream output_file2("/home/anamaria/Documents/mine.txt");
    std::ostream_iterator<float>output_iterator2(output_file2,"\n");
    std::copy(std::begin(mins),std::end(mins),output_iterator2);

    std::ofstream output_file3("/home/anamaria/Documents/maxe.txt");
    std::ostream_iterator<float>output_iterator3(output_file3,"\n");
    std::copy(std::begin(maxs),std::end(maxs),output_iterator3);


    std::ofstream output_file4("/home/anamaria/Documents/surfacecovered.txt");
    std::ostream_iterator<float>output_iterator4(output_file4,"\n");
    std::copy(std::begin(surfacecovered),std::end(surfacecovered),output_iterator4);

    std::ofstream output_file5("/home/anamaria/Documents/n_evo.txt");
    std::ostream_iterator<float>output_iterator5(output_file5,"\n");
    std::copy(std::begin(vertices_n),std::end(vertices_n),output_iterator5);



    /*
     * Printing all the values of means

    int size = sizeof(means) / sizeof(means[0]);

    // Print the array elements
    for (int i = 0; i < size; i++) {
        std::cout << means[i] << " ";
    }
    std::cout << std::endl;
    */
    return 0;
}

