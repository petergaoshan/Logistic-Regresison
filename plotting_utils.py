'''
Obtained this function from class project
'''
# Standard Imports
import numpy as np
import matplotlib.pyplot as plt


def plotter_classifier(w,basis_func, x, y, title=None, axis=None, grid_density=1000):
	'''
		Adopted from the mltools packaged created by Alex Ihler.

		Plots a 2D linear classifier along with the decision boundaries
		
		Parameters:
			w: The weights of the linear classifier
			basis_func: A python function that can create the basis function for the data points
			x: data to plot
			y: Classes of x
			title: A title to give the plot
			axis: a matplotlib axis.  If one is not provided then "plt" will be used
			grid_density: How dense of a grid should be used for rendering the decision boundary
	'''
	# Makes sure the data is only 2D
	if (x.shape[1] != 2):
		raise ValueError('plotter_classifier: function can only be called using two-dimensional data (features)')

	# Define an axis if one isnt passed in
	if(axis == None): 
		axis = plt 
	# axis.hold(True)
	axis.plot(x[:,0],x[:,1], color="black", visible=False )

	# TODO: can probably replace with final dot plot and use transparency for image (?)
	ax = axis.axis()
	xticks = np.linspace(ax[0],ax[1],grid_density)
	yticks = np.linspace(ax[2],ax[3],grid_density)
	grid = np.meshgrid( xticks, yticks )
	x_grid = np.column_stack( (grid[0].flatten(), grid[1].flatten()) )

	# Plot the colors
	grid_phi = basis_func(x_grid)
	y_hat_grid = np.argmax(np.matmul(grid_phi, w), 1)
	axis.imshow( y_hat_grid.reshape( (len(xticks),len(yticks)) ), extent=ax, interpolation='nearest',origin='lower',alpha=0.5, aspect='auto' )

	cmap = plt.cm.get_cmap()

	classes = np.unique(y)

	cvals = (classes - np.min(classes)) / (np.max(classes)-np.min(classes)+1e-100)
	for i,c in enumerate(classes): 
		axis.scatter( x[y==c,0],x[y==c,1], edgecolors="black", color=cmap(cvals[i]))  
	axis.axis(ax)

	if(title is not None):
		axis.title(title)