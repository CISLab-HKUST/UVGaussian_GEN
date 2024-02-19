
def c2f_img_size(iter, total_iter, down_size, up_size, ratio=0.5):
    iter_activate = int(total_iter * ratio)

    if iter >= iter_activate:
        return up_size

    iter_scale = iter / total_iter
    size = int(down_size + iter_scale * (up_size - down_size))
    
    return size