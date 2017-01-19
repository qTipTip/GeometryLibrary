# GeometryLibrary
A small Python(3) library for computing and visualizing geometric structures.
The library was written during a course on geometric modelling, and currently supports
mainly Bézier related objects.

The library offers the possibility of plotting and composing curves and surfaces in a
fairly small number of lines of code.

### Example

As a basic example, we want to interpolate a set of data points in 3D with C^2
continuity.

```python
import GeometryLibrary as gl

interpol_points = [(261, 703), (261, 738), (283, 718), (287, 723), (280, 735), (291, 753)]

f = gl.HermiteInterpolant(interpol_points, order_of_continuity=2, label='$f(t)$')
f.plot(display=True)
```
which produces the interpolant as shown in the image. 

![](http://i.imgur.com/th6oaGM.png)

The control points have been suitably chosen to interpolate with a C^2 continuity, and the control polygon is
visualized with small blue dots and grey lines. The big blue dots are the original data points. 
