import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_correspondences(images, uv_a, uv_b, use_previous_plot=None, circ_color='g', show=True):
    if use_previous_plot is None:
        fig, axes = plt.subplots(nrows=2, ncols=2)
    else:
        fig, axes = use_previous_plot[0], use_previous_plot[1]

    fig.set_figheight(10)
    fig.set_figwidth(15)
    pixel_locs = [uv_a, uv_b, uv_a, uv_b]
    axes = axes.flat[0:]
    if use_previous_plot is not None:
        axes = [axes[1], axes[3]]
        images = [images[1], images[3]]
        pixel_locs = [pixel_locs[1], pixel_locs[3]]
    for ax, img, pixel_loc in zip(axes[0:], images, pixel_locs):
        ax.set_aspect('equal')
        if isinstance(pixel_loc[0], int) or isinstance(pixel_loc[0], float):
            circ = Circle(pixel_loc, radius=10, facecolor=circ_color, edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
            ax.add_patch(circ)
        else:
            for x,y in zip(pixel_loc[0],pixel_loc[1]):
                circ = Circle((x,y), radius=10, facecolor=circ_color, edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
                ax.add_patch(circ)
        ax.imshow(img)
    if show:
        plt.show()
        return None
    else:
        return fig, axes

def plot_correspondences_from_dir(log_dir, img_a, img_b, uv_a, uv_b, use_previous_plot=None, circ_color='g', show=True):
    img1_filename = log_dir+"/images/"+img_a+"_rgb.png"
    img2_filename = log_dir+"/images/"+img_b+"_rgb.png"
    img1_depth_filename = log_dir+"/images/"+img_a+"_depth.png"
    img2_depth_filename = log_dir+"/images/"+img_b+"_depth.png"
    images = [img1_filename, img2_filename, img1_depth_filename, img2_depth_filename]
    images = [mpimg.imread(x) for x in images]
    return plot_correspondences(images, uv_a, uv_b, use_previous_plot=use_previous_plot, circ_color=circ_color, show=show)

def plot_correspondences_direct(img_a_rgb, img_a_depth, img_b_rgb, img_b_depth, uv_a, uv_b, use_previous_plot=None, circ_color='g', show=True):
    """

    Plots rgb and depth image pair along with circles at pixel locations
    :param img_a_rgb: PIL.Image.Image
    :param img_a_depth: PIL.Image.Image
    :param img_b_rgb: PIL.Image.Image
    :param img_b_depth: PIL.Image.Image
    :param uv_a: (u,v) pixel location, or list of pixel locations
    :param uv_b: (u,v) pixel location, or list of pixel locations
    :param use_previous_plot:
    :param circ_color: str
    :param show:
    :return:
    """
    images = [img_a_rgb, img_b_rgb, img_a_depth, img_b_depth]
    return plot_correspondences(images, uv_a, uv_b, use_previous_plot=use_previous_plot, circ_color=circ_color, show=show)
    
