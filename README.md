<div align="center">

# SKEIN
### Yarn Diameter Comparator

**A Streamlit research prototype for image-based yarn-width comparison.**

[Run locally](#1-run-skein-on-a-windows-pc) · [Present on a phone](#3-present-skein-from-a-phone-for-free) · [Update from GitHub](#4-keep-the-project-up-to-date)

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20prototype-8B8DFF)

</div>

---

## What is Skein?

Skein lets a user upload two yarn photographs, segment the apparent yarn body, estimate its image-based width, inspect the detected boundaries, and download a comparison report. The current interface uses the **Midnight Neon** theme and animated visual elements.

### Current features

- Two-image upload and side-by-side review.
- Otsu-based yarn segmentation, with optional experimental alternatives.
- Estimated width statistics and comparison screening.
- Calibration-target generation and camera-scale workflow.
- Boundary overlays, image-quality warnings, and downloadable reports.
- Automated software tests.

### Important measurement limitation

**Skein is a research prototype, not a validated measuring instrument.** Its ability to reliably detect a **0.017 mm (17 μm)** diameter difference has **not** been demonstrated on real yarn samples. A passing software test or a plausible-looking overlay does not establish physical accuracy. Validation requires a suitable physical calibration and independent reference measurements on real samples.

Do not treat uncalibrated pixel values as millimetres. Respect `INCONCLUSIVE` results and all quality warnings.

---

## 1. Run Skein on a Windows PC

These steps are for a new Windows computer. An internet connection is needed to clone the repository and install packages.

### Before you start

Install:

- **Git for Windows:** <https://git-scm.com/download/win>
- **Python:** install a recent compatible Python version from <https://www.python.org/downloads/>. During setup, enable **Add Python to PATH**.
- **VS Code** (recommended): <https://code.visualstudio.com/download>

### First-time setup

1. Open **VS Code**.
2. Choose **Terminal → New Terminal**. Make sure the terminal is PowerShell.
3. Move to the folder where you want the project, then clone it:

   ```powershell
   cd "$HOME\Desktop"
   git clone https://github.com/mohammednaeemog-sys/skein-yarn-comparator.git
   cd skein-yarn-comparator
   ```

   **If the repository is private:** the repository owner must first invite your GitHub account as a collaborator. Sign in through Git when prompted; never put a GitHub password or access token in this README or in source code.

4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use a temporary policy for this terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

5. Install the main dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. Launch the Midnight Neon app:

   ```powershell
   python -m streamlit run yarn_diameter_app_neon.py
   ```

7. Streamlit will print a local address, usually **https://skein-yarn-studio.streamlit.app/**. Open it in the browser on that PC.

To stop the app, focus the terminal and press **Ctrl+C**.

### Optional: run the tests

With the virtual environment activated and the terminal inside the repository folder:

```powershell
python -m py_compile yarn_diameter_app_neon.py yarn_measurement.py yarn_calibration.py yarn_ml.py
python -m pytest -v
```

The project previously had 15 passing tests. Re-run them after changes; results from an earlier run are not a guarantee that later code still passes.

---

## 2. Basic use

1. Open Skein.
2. Upload **two different yarn photos**—one for Sample A and one for Sample B.
3. Use the same camera, lens, zoom, focus, distance, resolution, orientation, and lighting for both captures. Keep one strand straight and leave visible background around its width-defining edges.
4. Under **Advanced image settings**, start with:
   - **Boundary detection:** Otsu threshold.
   - **Yarn/background contrast:** select the option matching the photograph. For dark yarn on a light background, choose **Dark yarn on light background**.
   - **Correct uneven illumination:** Off by default.
   - **3×3 median filter:** Off by default; our pilot tests showed its effect on estimated width was inconsistent.
5. Inspect the green mask. It should follow the main yarn body, not the background or loose hairs. If the mask is wrong or warnings appear, do not rely on the estimate.
6. For millimetre estimates, complete the calibration workflow using the calibration target printed at **actual size** and photographed at the same plane as the yarn. Follow the app's instructions.
7. Review the result, quality warnings, and downloaded report. If the app says `INCONCLUSIVE`, do not override that by treating the values as confirmed measurements.

A calibration profile is local and is intentionally excluded from GitHub. Each computer/camera setup must be calibrated separately.

---

## 3. Present Skein from a phone for free

The easiest way to show Skein when a laptop is not available is to deploy it to **Streamlit Community Cloud** and open the resulting web link on the phone. Community Cloud currently offers free deployment, subject to its service limits and terms. See the official documentation: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>.

### A. Deploy once

1. Push the latest working Skein code to GitHub (see [Keep the project up to date](#4-keep-the-project-up-to-date)).
2. Open **Streamlit Community Cloud**: <https://share.streamlit.io/>.
3. Sign in with GitHub and authorize Streamlit to access the repository. If the repository is private, grant the required private-repository permission in Streamlit's linked-account settings.
4. Select **Create app** and choose **Yup, I have an app** (wording may vary slightly).
5. Choose:
   - **Repository:** `mohammednaeemog-sys/skein-yarn-comparator`
   - **Branch:** `main`
   - **Main file path:** `yarn_diameter_app_neon.py`
6. Choose an available app URL/subdomain if offered, then select **Deploy**.
7. Wait for the build to finish. If deployment fails, open the app's logs and check the dependency error before changing files. Streamlit uses `requirements.txt` to install Python dependencies: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies>.

### B. Open it on a phone

1. Copy the deployed `https://….streamlit.app` URL from the Streamlit workspace.
2. Send the URL to the phone, or open it in the phone's browser.
3. For an in-person presentation, display a QR code for that URL on a slide or printout. Scan it with the phone camera; on most phones this opens the link in the browser.
4. Keep a few **non-sensitive demo photos** on the phone in case you want to demonstrate the upload flow. Do not use private client or research images in a public demo.

After the initial deployment, the presentation can be opened from the phone without having the development PC present. Internet access is required, and the free service can have usage/resource limits or take time to wake up.

### Very important: cloud calibration caveat

The current application stores its calibration profile in a file next to the app. On a hosted app, that file is not a per-user calibration system and should not be assumed to persist through service restarts or redeployments. A shared hosted app can also be used by people with different cameras. **Treat the hosted version as a presentation/workflow demo only until calibration storage is changed to be per-session or per-user and revalidated. Do not use its cloud measurements for production decisions or claim 17 μm accuracy.**

If you need to demonstrate physical measurements, use the locally calibrated app on the machine/camera it was calibrated for until the hosted calibration behavior has been corrected and tested.

### Private source, shareable demo

You can keep the GitHub repository private and authorize Streamlit to access it. The deployed app may be made public in Streamlit's app settings if you want anyone with the URL to open it. If the app is left private, add the intended viewers in Streamlit. See: <https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app>.

---

## 4. Keep the project up to date

GitHub is the central copy of the source code. **Saving a file in VS Code does not send it to GitHub by itself.** Updates happen when you pull, commit, and push.

### On the PC where you edit Skein

1. Open the project folder in VS Code.
2. Pull any updates first:

   ```powershell
   git pull --ff-only
   ```

3. Edit and save your files.
4. Run tests before publishing:

   ```powershell
   python -m pytest -v
   ```

5. Stage the files you deliberately want to publish, then commit and push. Example for a change to the main app:

   ```powershell
   git add yarn_diameter_app_neon.py
   git commit -m "Improve Skein interface"
   git push
   ```

   Replace the file name and commit message to match your change. For several intentional changes, stage each relevant file by name. Avoid `git add .` until you have checked `git status` and confirmed that no photos, datasets, calibration profiles, or model checkpoints will be uploaded.

### On another PC / the client's PC

If the repo is already cloned, open a terminal in the repository folder and run:

```powershell
 git pull --ff-only
```

Then activate the virtual environment and restart Skein. If the client has not cloned it yet, follow the first-time setup in Section 1.

### Hosted app updates

When the app is connected to the GitHub repository, pushing a new commit to the configured branch normally triggers a new deployment automatically. Check the Streamlit workspace/logs if the live page does not update. Dependency changes in `requirements.txt` may take longer because the environment needs to be rebuilt: <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>.

### Optional sync helper

This repository includes `sync_skein_to_github.ps1`. If it is available in your clone, it can assist with committing and pushing changes to **already tracked files**:

```powershell
Unblock-File .\sync_skein_to_github.ps1
.\sync_skein_to_github.ps1 -CommitMessage "Describe your change"
```

If PowerShell still blocks it, do not weaken the machine-wide execution policy. Use the manual `git add`, `git commit`, and `git push` commands above. The helper intentionally does **not** automatically add new/untracked files; add a new intended file by name first.

---

## 5. Files and repository hygiene

- `.gitignore` excludes local datasets/photos, calibration profiles, reports, virtual environments, and model checkpoints.
- `calibration_profile.json` is local machine/setup data and should not be committed.
- Do not commit secrets, passwords, personal photos, client data, or credentials.
- `requirements.txt` lists the normal app dependencies. `requirements-ml.txt` is optional and is for additional ML/model training tools; it is not needed for a standard Otsu-based demo unless you intentionally use those features.
- The app's trained AI options are available only if compatible, trained checkpoints and required ML packages are installed. Do not assume that the current default app deployment contains a validated AI model.

---

## 6. Troubleshooting

| Problem | What to try |
|---|---|
| `git` is not a repository | Change directory into the folder containing `.git`, usually the cloned `skein-yarn-comparator` directory. |
| `requirements.txt` not found | Ensure the terminal is in the repository root, then run `Get-ChildItem`. |
| Streamlit command not found | Activate `.venv` and run `python -m pip install -r requirements.txt`. |
| App fails on Streamlit Community Cloud | Open deployment logs; verify `requirements.txt`, the entrypoint path, and compatible dependencies. |
| Calibration missing after restart | The current profile is local file-based and not a durable cloud/user calibration system. Recalibrate locally; do not assume the hosted profile persisted. |
| Result is `INCONCLUSIVE` | Check calibration, image dimensions, yarn/background polarity, boundary overlay, and quality warnings. Do not bypass a warning to force a result. |

---

## Project status

Skein is a functional research prototype. Fifteen software tests passed in a previous local run; re-run them on the current code before release. **The real-world ability to resolve a 0.017 mm yarn-diameter difference remains unvalidated.** This README describes how to run and demo the software; it is not a claim of metrology accuracy.

## Useful links

- **GitHub repository:** <https://github.com/mohammednaeemog-sys/skein-yarn-comparator>
- **Streamlit Community Cloud:** <https://share.streamlit.io/>
- **Streamlit deployment guide:** <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
- **Streamlit app sharing:** <https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app>

---
## Authors :
Mohammed Naeem
Jasim Uddin Shimul

<sub>Built as a research prototype. Confirm calibration and measurement validity before using results for textile quality-control or other consequential decisions.</sub>
