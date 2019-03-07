# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot(x,y):
	"""
		plot different graphs. 
	"""
	try: 
		plt.plot(x,y)
		plt.show()
	except Exception: 
		print("    Something went wrong! plot method did not work!\n")

#------------------------------------
# testing
if __name__ == "__main__":
    x = [1,0,1,0,1,0,1,0,1,0]
    y = [1,0,1,0,1,0,1,0,1,0]
    print("Trying plot()!\n")
    content = plot(x,y)
    if content != None:
        print("    plot() works.\n")

