from PIL import Image

def create_gif(imgs, fp_out, duration=200):
    img, *imgs = imgs
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=duration, loop=0, quality=20)
