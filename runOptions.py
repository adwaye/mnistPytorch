#cnn params

init_width = 32
#dataloader params
batch_size = 100
l_rate = 0.001
momentum = 0.0
weight_decay = 0.0
dampening = 0.0
nesterov = 0  #0 or 1
lr_gamma = 0.9
epochs = 100
#transforms params
trans_probability = 0.5
max_rotation_right = 9
max_rotation_left = 9
max_factor = 1.15
min_factor = 1.05
grid_height = 5
grid_width = 5
magnitude = 3
brightness = 2
contrast = 2
translate = 0.1
blur_max = 3
blur_min = 0.5
blur_kernel = 5
opt_kwargs = dict(batch_size=batch_size,
                  l_rate=l_rate,
                  lr_gamma=lr_gamma,
                  momentum=momentum,
                  weight_decay=weight_decay,
                  dampening=dampening,
                  nesterov=nesterov,
                  probability=trans_probability,
                  max_rotation_right=max_rotation_right,
                  max_rotation_left=max_rotation_left,
                  max_factor=max_factor,
                  min_factor=min_factor,
                  grid_height=grid_height,
                  grid_width=grid_width,
                  magnitude=magnitude,
                  init_width=init_width,
                  brightness=brightness,
                  contrast=contrast,
                  translate=translate,
                  blur_kernel=blur_kernel,
                  blur_min=blur_min,
                  blur_max=blur_max,

                  )
optimizer = 'adam'