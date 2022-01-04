from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


if __name__ == '__main__':
    load_path = r'\dataset\train'
    ori_path = os.path.join(load_path, 'train')
    name_ori = [name for name in sorted(os.listdir(ori_path))
                if len(name.split('_')) == 3][0]
    name = [name for name in
            sorted(os.listdir(os.path.join(load_path, 'train_mask')))][0]

    tif = TIFF.open(os.path.join(load_path, 'train', name_ori), mode='r')
    image_ori = tif.read_image()
    tif = TIFF.open(os.path.join(load_path, 'train_mask', name), mode='r')
    image = tif.read_image()

    cmap = 'jet'
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(image/255., cmap=cmap)
    plt.axis('off')
    np.unique(image)
    np.unique(image_ori)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.figure(figsize=(8, 6))
    image_ori_ = Resize((448, 576))(Image.fromarray(image_ori))
    image_ori_ = np.array(image_ori_)/255.
    ax = plt.gca()
    im = ax.imshow(image_ori_, cmap=cmap)
    plt.axis('off')
    np.unique(image_ori)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
