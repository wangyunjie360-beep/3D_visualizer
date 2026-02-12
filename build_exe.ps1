conda activate wyj

pip install -r requirements.txt

pyinstaller pcd_viewer_app.py `
  --noconsole `
  --name PointCloudViewer `
  --onedir `
  --clean `
  --collect-all open3d `
  --collect-all tkinterdnd2
