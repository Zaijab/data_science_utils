def plot_figure_from_filepath(filepath):
    import matplotlib.image as mpimg
    image = mpimg.imread(filepath)
    plt.figure(figsize = (12,12))
    plt.gca().set_axis_off()
    plt.imshow(image)
    plt.show()
