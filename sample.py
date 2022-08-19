from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
from IPython import display
import geopandas as gpd
import pandas as pd
import numpy as np

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.stream = self.data_stream
        self.anno = []
        # Setup the figure and axes...
        self.fig, (self.ax, self.cax) = plt.subplots(1,2, gridspec_kw={'width_ratios':[50,1]})
        # set up colorbar
        self.scat = self.ax.scatter([], [])
        bounds = [0, 10, 30, 50, 100, 200]

        cmap = matplotlib.cm.hot
        self.norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        colorbar = self.fig.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=cmap),
                                    cax=self.cax, orientation='vertical',
                                    boundaries = bounds,
                                    ax = self.ax,
                                    label="Discrete intervals with extend='both' keyword",
                                    extend='both',  
                                    ticks=bounds,
                                    spacing='proportional')

        # colorbar = self.fig.colorbar(self.scat, ax = self.ax ,label = 'log&10%(flation) /%')


        # colorbar.set_ticklabels([r'$<10^{0}$', 1, 2, r'$10^{14}$', r'$10^{14}+12345678$'])
        # colorbar.set_label(r'$n_e$ in $m^{-3}$', labelpad=-40, y=0.45)

        countries = gpd.read_file(
                gpd.datasets.get_path("naturalearth_lowres"))
        countries.plot(ax = self.ax)

        # Then setup FuncAnimation.
        self.ani = FuncAnimation(self.fig, self.update,frames = 48, interval=200, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        
        colors = self.dataframe[str(1973)]
        colors = colors.to_numpy().astype(float)

        colors = self.stream(0)

        self.scat = self.ax.scatter(self.dataframe['Longitude'], self.dataframe['Latitude'],
                        c = colors, cmap = 'hot', norm = self.norm)

        


        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, t):

        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        colors = self.dataframe[str(t + 1973)]
        colors = colors.to_numpy().astype(float)
        return colors
        

    def update(self, i):
        """Update the scatter plot."""

        c = self.stream(i)

        self.ax.set_title('Flation of world ' + str(i + 1973))

        # Set colors..
        self.scat.set_array(c)

        # hiện random 20 nước
        np.random.seed(10)
        row = np.round((np.random.random(20)*1000)%134)
        row = row.astype(np.int16)

        # remove hết anno trước khi tạo mới
        for x in self.anno:
            x.remove()

        self.anno = []

        for temp in row:
            country = self.dataframe.iloc[temp]
            flat = round(country[str(i + 1973)], 2)
            anno = f'{country["Country Name"]}: {flat}%'
            # anno = f'{country["Country Name"]}: {country[str(temp)]}%'
            anno_picec = self.ax.annotate(anno,
                    xy=(country['Longitude'], country['Latitude']), xycoords='data',
                    xytext= (country['Longitude'] - 0.35, country['Latitude'] + 0.2),
                )
            self.anno.append(anno_picec)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat

data_world_dataframe = pd.read_csv('out.csv')
a = AnimatedScatter(data_world_dataframe)
plt.show()
