# GitHub Pages Site

This directory contains the GitHub Pages website for the PALMS & PALMS+ project.

## Setup Instructions

To enable GitHub Pages for this repository:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select:
   - **Deploy from a branch**
   - **Branch**: `main` (or `master`)
   - **Folder**: `/docs`
4. Click **Save**

The site will be available at:
`https://[your-username].github.io/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones/`

Or if using a custom domain, configure it in the Pages settings.

## File Structure

```
docs/
├── index.html          # Main HTML page
├── assets/
│   ├── css/
│   │   └── style.css   # Stylesheet
│   └── js/
│       └── main.js     # JavaScript for interactivity
└── images/             # Project images and visualizations
```

## Local Testing

To test the site locally before deploying:

1. Use a local web server (Python example):
   ```bash
   cd docs
   python -m http.server 8000
   ```
2. Open `http://localhost:8000` in your browser

Or use any other local web server tool like `live-server`, `http-server`, etc.

