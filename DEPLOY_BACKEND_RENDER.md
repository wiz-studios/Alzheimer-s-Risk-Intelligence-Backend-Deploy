# Deploy Backend For Native Android App

This folder is the public backend required by the native Android app.

The Android APK cannot work for a remote client if the backend only runs on your laptop. Deploy this backend first, then rebuild the Android app using the hosted backend URL.

## Recommended Platform

Use Render because this backend is a Python web service and includes model/data files.

## Deploy Steps

1. Create a new GitHub repository for this backend.
2. Upload the contents of this folder to that repository.
3. Go to Render.
4. Create a new `Web Service`.
5. Connect the GitHub repository.
6. Use these settings:

```text
Environment: Python
Build command: pip install -r requirements.txt
Start command: python mobile_backend.py
Health check path: /api/health
```

7. Deploy.

## Test After Deployment

Open:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api/health
```

Expected response:

```json
{"status":"ready"}
```

Then test:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api/datasets
```

Warm the default model before client demos:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api/warmup?dataset=data1.csv&model=xgboost
```

The Android app also calls warmup automatically after loading the selected dataset/model.

## GitHub Upload Note

This full backend keeps v5 functionality, including the larger `data1_hybrid_voting.joblib` model. That file is below GitHub's 100 MB hard limit, but GitHub's browser upload page may reject files above 25 MB.

If browser upload fails, use one of these instead:

- GitHub Desktop
- command-line `git add`, `git commit`, `git push`
- Git LFS

Do not remove model files if the goal is to preserve the full v5 system.

## Next Step After Hosting

Send the deployed backend URL back to Codex. Then update the Android app:

```text
android_app/app/src/main/java/com/partb/alzheimersriskintelligence/MainActivity.java
```

from:

```text
http://10.0.2.2:8510/api
```

to:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api
```

After that, build the APK in Android Studio and email it to the client.

## Important

This backend is for an academic decision-support prototype. It is not a clinical diagnostic service.
