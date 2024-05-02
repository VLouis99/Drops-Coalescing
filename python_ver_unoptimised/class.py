from functions import *



class drop:
	def __init__(self,index,point,radius):
		self.index = index
		self.point = point
		self.radius = radius
	
	def __str__(self):
		return f"{self.index}({self.point}){self.radius}"
	
	
	def radius_calc(self,t):
		return self.radius * t
