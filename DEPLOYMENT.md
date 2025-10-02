# 🏏 Cricket Win Predictor - Vercel Deployment Guide

## Quick Deploy to Vercel

### Option 1: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy the project**:
   ```bash
   vercel
   ```

4. **Follow the prompts**:
   - Set up and deploy? **Yes**
   - Which scope? **Your account**
   - Link to existing project? **No**
   - Project name: **cricket-win-predictor** (or your preferred name)
   - Directory: **./** (current directory)

### Option 2: Deploy via GitHub + Vercel Dashboard

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/cricket-win-predictor.git
   git push -u origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Deploy!

## 📁 Project Structure

```
cricket-win-predictor/
├── app.py                 # Main Streamlit application
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── DEPLOYMENT.md        # This file
└── README.md            # Project documentation
```

## 🚀 Features

- **Progressive Interface**: Step-by-step match setup
- **Advanced Analytics**: Multi-factor prediction algorithm
- **Real-time Predictions**: Head-to-head, ground advantage, run rates
- **Responsive Design**: Works on desktop and mobile

## 🔧 Configuration

The `vercel.json` file is already configured for optimal deployment:
- Python runtime
- Proper routing
- Streamlit compatibility

## 📊 Prediction Algorithm

The app uses advanced analytics including:
- Team strengths and rankings
- Head-to-head historical data
- Ground-specific advantages
- Real-time match situation analysis
- Run rate and wicket impact calculations

## 🎯 Usage

1. Select teams and venue
2. Complete toss information
3. Enter match progress data
4. Get instant win probability predictions

## 🌐 Live Demo

Once deployed, your app will be available at:
`https://your-project-name.vercel.app`

## 🛠️ Troubleshooting

If deployment fails:
1. Check Python version compatibility
2. Verify all dependencies in `requirements.txt`
3. Ensure `vercel.json` configuration is correct
4. Check Vercel build logs for specific errors

## 📞 Support

For issues or questions about deployment, check:
- [Vercel Documentation](https://vercel.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
