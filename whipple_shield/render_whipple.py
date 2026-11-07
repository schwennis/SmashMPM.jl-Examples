from paraview.simple import *
import glob, os, sys

out_dir    = sys.argv[1] if len(sys.argv) > 1 else "output"
frames_dir = os.path.join(out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(out_dir, "whipple_shield*.xdmf")) +
               glob.glob(os.path.join(out_dir, "whipple_shield*.xmf")))
assert files, f"Keine XDMF-Datei in {out_dir} gefunden"

reader = XDMFReader(FileNames=files)
reader.UpdatePipelineInformation()
print("Arrays:   ", reader.PointData.keys())      # hier die echten Feldnamen nachschauen
print("Timesteps:", reader.TimestepValues)

# Anpassen an die Feldnamen aus der Ausgabe oben
COLOR_ARRAY = "velocity"        # z.B. velocity, pressure, damage, body_id ...
COLOR_COMP  = "Magnitude"       # bei Skalaren: None

# Halbschnitt bei y = 0, damit man in die Platten hineinsieht
clip = Clip(Input=reader)
clip.ClipType = "Plane"
clip.ClipType.Origin = [0.0, 0.0, 0.0]
clip.ClipType.Normal = [0.0, 1.0, 0.0]
clip.Invert = 0                 # ggf. auf 1 setzen, falls die falsche Hälfte bleibt

view = CreateView("RenderView")
view.ViewSize = [1920, 1080]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]

disp = Show(clip, view)
disp.SetRepresentationType("Point Gaussian")   # alternativ: "Points" mit PointSize
disp.GaussianRadius = 0.00015                  # ~ DX/2, in m
disp.ShaderPreset = "Sphere"
if COLOR_COMP:
    ColorBy(disp, ("POINTS", COLOR_ARRAY, COLOR_COMP))
else:
    ColorBy(disp, ("POINTS", COLOR_ARRAY))

times = reader.TimestepValues
if not hasattr(times, "__len__"):
    times = [times]

# Farbskala über alle Zeitschritte fixieren (Min/Max vom letzten Schritt reicht meist nicht,
# daher ggf. fest vorgeben, z.B. lut.RescaleTransferFunction(0, 8000))
view.ViewTime = times[-1]
disp.RescaleTransferFunctionToDataRange(False, True)
lut = GetColorTransferFunction(COLOR_ARRAY)
# lut.RescaleTransferFunction(0.0, 8000.0)
disp.SetScalarBarVisibility(view, True)

# Kamera: Blick entlang -y auf die x-z-Ebene, Einschlag läuft in +x
view.CameraPosition   = [0.02, -1.0, 0.0]
view.CameraFocalPoint = [0.02,  0.0, 0.0]
view.CameraViewUp     = [0.0, 0.0, 1.0]
view.ResetCamera()

for i, t in enumerate(times):
    view.ViewTime = t
    Render(view)
    SaveScreenshot(os.path.join(frames_dir, f"frame_{i:04d}.png"), view,
                   ImageResolution=[1920, 1080])
    print(f"Frame {i+1}/{len(times)}  t = {t:.3e} s", flush=True)