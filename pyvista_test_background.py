import pyvista as pv

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/pyvista_test_background.py
"""

import pyvista as pv

pl = pv.Plotter(lighting=None)

# Your 360-degree panorama; a normal non-panoramic JPEG will look stretched
texture = pv.read_texture("background_2.jpeg")

# Add it as the surrounding skybox
skybox = texture.to_skybox()
pl.add_actor(skybox)

# Optional: use it for lighting/reflections too
pl.set_environment_texture(texture, is_srgb=True)

# Object in the middle
pl.add_mesh(pv.Sphere(radius=0.5), color="white")

pl.camera_position = "xy"
pl.show()