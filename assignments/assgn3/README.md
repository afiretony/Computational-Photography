# Homework Assignment 3

15-663, Computational Photography, Fall 2022, Carnegie Mellon University

Chenhao Yang

---

## 1. Bilateral filtering

![](data/lamp/final.jpg)

In this section, we implemented four algorithms:

- Piecewise bilateral filtering 
- Joint flash / no-flash bilateral filtering
- Detail transfer
- Shadow and specularity mask

The first part **Parameters Evaluation** will present effects of various parameters used in bilateral filtering, as they are essential for noise removal and general quality of the filtered images. The second part **Algorithm Evaluation** will compare the four different algorithms and analyse their pros / cons.

### 1.1 Parameters Evaluation

#### $$\sigma_r$$

$$\sigma_r$$  is the standard deviation of the intensity Gaussian kernel.

| value | Image produced                  | Difference                        |
| ----- | ------------------------------- | --------------------------------- |
| 0.08  | ![](data/lamp/sigma_r/0.08.jpg) | ![](data/lamp/sigma_r/D_0.08.jpg) |
| 0.10  | ![](data/lamp/sigma_r/0.1.jpg)  | ![](data/lamp/sigma_r/D_0.1.jpg)  |
| 0.12  | ![](data/lamp/sigma_r/0.12.jpg) | ![](data/lamp/sigma_r/D_0.12.jpg) |
| 0.14  | ![](data/lamp/sigma_r/0.14.jpg) | ![](data/lamp/sigma_r/D_0.14.jpg) |

Smaller $$\sigma_r$$ makes edges clearer because the spacial weighting is more concentrated, nearby pixels weights more and farby pixels weights less, thus makes edges clearer.



#### $$\sigma_s$$

$$\sigma_s$$  is the standard deviation of the spatial Gaussian kernel.

| value | Image produced                  | Difference                        |
| ----- | ------------------------------- | --------------------------------- |
| 0.1   | ![](data/lamp/sigma_s/0.1.jpg)  | ![](data/lamp/sigma_s/D_0.1.jpg)  |
| 1.0   | ![](data/lamp/sigma_s/1.0.jpg)  | ![](data/lamp/sigma_s/D_1.0.jpg)  |
| 5.0   | ![](data/lamp/sigma_s/5.0.jpg)  | ![](data/lamp/sigma_s/D_5.0.jpg)  |
| 10.0  | ![](data/lamp/sigma_s/10.0.jpg) | ![](data/lamp/sigma_s/D_10.0.jpg) |

Larger $$\sigma_s$$  removes more noise because the "averaging" effects of gaussian kernel, it is noticable from the noise density from difference maps. However, larger $$\sigma_s$$ washes out details of the image and it can be considered as a drawback. Another fact is that the smoothing effect is limited and saturates when $$\sigma_s$$ reached some level as we cannot notice significant difference between `sigma_s=50` and `sigma_s=10`.



#### $$kernel$$

$$kernel$$ is the kernel size of gaussian filter applied.

| value | Image produced               | Difference                     |
| ----- | ---------------------------- | ------------------------------ |
| 5     | ![](data/lamp/kernel/5.jpg)  | ![](data/lamp/kernel/D_5.jpg)  |
| 9     | ![](data/lamp/kernel/9.jpg)  | ![](data/lamp/kernel/D_9.jpg)  |
| 15    | ![](data/lamp/kernel/15.jpg) | ![](data/lamp/kernel/D_15.jpg) |
| 29    | ![](data/lamp/kernel/29.jpg) | ![](data/lamp/kernel/D_29.jpg) |

Larger kernel smoothes out the image better, however it washes out details of the image as well. 



### 1.2 Algorithm Evaluation

| Type                        | Image produced               | Difference                     |
| --------------------------- | ---------------------------- | ------------------------------ |
| Piecewise Bilateral         | ![](data/lamp/piecewise.jpg) | ![](data/lamp/D_piecewise.jpg) |
| Joint Bilateral             | ![](data/lamp/joint.jpg)     | ![](data/lamp/D_joint.jpg)     |
| Detail Transfer             | ![](data/lamp/detail.jpg)    | ![](data/lamp/D_detail.jpg)    |
| Shadow and specularity Mask | ![](data/lamp/final.jpg)     | ![](data/lamp/D_final.jpg)     |

*The difference map is computed by `normalize(image[i] - image[i-1])`, e.g., difference map for joint bilateral is `image[joint bilateral]-image[piecewise bilateral]`. We believe this makes more sense when comparing two algorithms.



Here are finds of four algorithms:

- Piecewise bilateral filtering 

  Advantage: Piecewise bilateral filter removed most of the noises in the scene, especially dark areas where originally contain much noises. 

  Disadvantage: removed details of the image

  

- Joint flash / no-flash bilateral filtering

  Advantage: Details are more significant

  Disadvantage: Allignement between flash / no-flash pairs has to be established; specularities and shadows are not removed

  

- Detail transfer

  Advantage: details are recovered while removing noises

  Disadvantage: Not yet considered shadow and specularity contamination

  

- Shadow and specularity mask

  Advantage: can be considered as optimized solution of above

  Disadvantage: need manual inspection and selection of mask threshold



## 2. Gradient-domain processing

![](data/museum/i_ambient_b_flash.jpg)

### 2.1 Differentiate and then re-integrate an image

Implemented using both gradient and Laplacian filtering, re-integrated images with no artifacts.

### 2.2 Create the fused gradient field

Gradient fields of ambient image:

| $$I_x$$                    | $$I_y$$                    |
| -------------------------- | -------------------------- |
| ![](data/museum/amb_x.jpg) | ![](data/museum/amb_y.jpg) |



Gradient fields of flash image:

| $$I_x$$                    | $$I_y$$                    |
| -------------------------- | -------------------------- |
| ![](data/museum/fls_x.jpg) | ![](data/museum/fls_y.jpg) |



Gradient fields of fused image:

| $$I_x$$                     | $$I_y$$                     |
| --------------------------- | --------------------------- |
| ![](data/museum/grad_x.jpg) | ![](data/museum/grad_y.jpg) |

Parameters:

| $$\tau_s$$ | $$\sigma$$ | $$\epsilon$$ | N    |
| ---------- | ---------- | ------------ | ---- |
| 1.0        | 40         | 0.001        | 1000 |

With different boundary conditions:

|                | I_bound=ambient                          | I_bound=flash                          | I_bound=average                          |
| -------------- | ---------------------------------------- | -------------------------------------- | ---------------------------------------- |
| I_init=ambient | ![](data/museum/i_ambient_b_ambient.jpg) | ![](data/museum/i_ambient_b_flash.jpg) | ![](data/museum/i_ambient_b_average.jpg) |
| I_init=flash   | ![](data/museum/i_flash_b_ambient.jpg)   | ![](data/museum/i_flash_b_flash.jpg)   | ![](data/museum/i_flash_b_average.jpg)   |
| I_init=average | ![](data/museum/i_average_b_ambient.jpg) | ![](data/museum/i_average_b_flash.jpg) | ![](data/museum/i_average_b_average.jpg) |
| I_init=zeros   | ![](data/museum/i_zeros_b_ambient.jpg)   | ![](data/museum/i_zeros_b_flash.jpg)   | ![](data/museum/i_zeros_b_average.jpg)   |

The initialization doesn't matter: the poission solver will converge to desired results from any initialization. However, good initialization helps computation time.

The boundary condition will be different, in our case boundary with flash is slightly brighter than ambient boundary, and average is in between.



## 3. Capture your own flash/no-flash pairs

### 3.1 Bilateral filtering

Captured image pair:

| Ambient                               | Flash                               |
| ------------------------------------- | ----------------------------------- |
| ![](data/kitchen/kitchen_ambient.jpg) | ![](data/kitchen/kitchen_flash.jpg) |

Fused result:

![](data/kitchen/final.jpg)

To reproduce:

```bash
python bilateral_filtering.py --scene 'kitchen' --kernel 5
```



### 3.2 Gradient domain image fusion

Captured image pair:

| Ambient                           | Flash                           |
| --------------------------------- | ------------------------------- |
| ![](data/glass/glass_ambient.jpg) | ![](data/glass/glass_flash.jpg) |

Fused:

![](data/glass/res.jpg)
