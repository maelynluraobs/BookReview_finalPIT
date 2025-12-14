# Deployment Guide: Amazon Book Review Analyzer on Render

## 🚀 Complete Deployment Steps

### Prerequisites
1. GitHub account
2. Render account (free tier available at https://render.com)
3. Git installed on your computer

---

## Step 1: Prepare Your Repository

### 1.1 Initialize Git Repository (if not already done)
```bash
cd C:\Users\crstn\bookreview
git init
```

### 1.2 Add All Files
```bash
git add .
```

### 1.3 Commit Files
```bash
git commit -m "Initial commit - Amazon Book Review Analyzer"
```

### 1.4 Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `amazon-book-review-analyzer`
3. Make it **Public** (required for free Render deployment)
4. Don't initialize with README (you already have files)
5. Click "Create repository"

### 1.5 Push to GitHub
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/amazon-book-review-analyzer.git
git push -u origin main
```

**Replace `YOUR_USERNAME` with your GitHub username**

---

## Step 2: Deploy on Render

### 2.1 Sign Up / Log In to Render
1. Go to https://render.com
2. Sign up or log in (can use GitHub account)

### 2.2 Create New Web Service
1. Click **"New +"** button
2. Select **"Web Service"**

### 2.3 Connect Repository
1. Click **"Connect a repository"**
2. Authorize Render to access your GitHub
3. Select: `amazon-book-review-analyzer` repository

### 2.4 Configure Service

**Basic Settings:**
- **Name:** `book-review-analyzer` (or any name you prefer)
- **Region:** Oregon (or closest to you)
- **Branch:** `main`
- **Root Directory:** (leave empty)
- **Runtime:** `Python 3`

**Build Settings:**
- **Build Command:** 
  ```bash
  pip install -r requirements.txt && python train_and_save_model.py
  ```

- **Start Command:**
  ```bash
  gunicorn app:app
  ```

**Instance Type:**
- Select **Free** tier

### 2.5 Environment Variables (Optional)
Click **"Advanced"** and add:
- Key: `PYTHON_VERSION`, Value: `3.11.0`
- Key: `NLTK_DATA`, Value: `/opt/render/project/src/nltk_data`

### 2.6 Deploy
1. Click **"Create Web Service"**
2. Wait for deployment (5-10 minutes for first build)
3. Watch the build logs

---

## Step 3: Deployment Process

### What Happens During Deployment:

1. **Install Dependencies** (~2 min)
   - Installs Flask, scikit-learn, NLTK, pandas, etc.

2. **Download NLTK Data** (~1 min)
   - Downloads stopwords and wordnet

3. **Train Model** (~3-5 min)
   - Loads Books_ratings.csv (29,352 reviews)
   - Trains Logistic Regression model
   - Saves sentiment_model.pkl and tfidf_vectorizer.pkl

4. **Start Server** (~30 sec)
   - Starts Gunicorn server
   - Loads model files
   - Ready to serve requests!

---

## Step 4: Access Your Deployed Application

### Your Live URL
After successful deployment, you'll get a URL like:
```
https://book-review-analyzer.onrender.com
```

### Test the Application:
1. **Homepage:** Visit the URL directly
2. **Generate Review:** Click "Generate Review" button
3. **Analyze Sentiment:** Click "Analyze Sentiment"
4. **About Page:** Click "About" to view project info

---

## Step 5: Verify Deployment

### Check API Endpoints:

**1. Predict Sentiment:**
```bash
curl -X POST https://book-review-analyzer.onrender.com/predict_sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This book was amazing!"}'
```

**2. Get Random Review:**
```bash
curl https://book-review-analyzer.onrender.com/get_random_review
```

**3. Homepage:**
Open in browser: `https://book-review-analyzer.onrender.com`

---

## Important Notes

### ⚠️ Free Tier Limitations:
- **Spin down after inactivity:** Service sleeps after 15 minutes of no requests
- **First request slow:** Takes 30-60 seconds to wake up
- **750 hours/month:** Free tier limit
- **No persistent disk:** Files are recreated on each deployment

### ✅ What's Included:
- ✅ Complete Flask backend
- ✅ Frontend HTML interface
- ✅ ML model training on deployment
- ✅ Dataset (Books_ratings.csv)
- ✅ All 29,352 book reviews
- ✅ Jupyter notebook (accessible via repo)

### 📁 Deployed Files:
```
amazon-book-review-analyzer/
├── app.py                          # Flask server
├── index.html                      # Frontend UI
├── train_and_save_model.py         # Model training script
├── Books_ratings.csv               # Dataset
├── book_review_dataset.ipynb       # Jupyter notebook
├── requirements.txt                # Dependencies
├── render.yaml                     # Render config
├── .gitignore                      # Git ignore rules
├── README.md                       # Project README
└── section-Obejero-Llatuna.md     # Documentation
```

---

## Troubleshooting

### Issue 1: Build Fails
**Solution:** Check build logs for errors
- Ensure all files are committed to GitHub
- Verify requirements.txt is correct
- Check if Books_ratings.csv uploaded (GitHub has 100MB file limit)

### Issue 2: Application Not Loading
**Solution:** 
- Wait 60 seconds for cold start
- Check Render logs for errors
- Verify PORT environment variable is set

### Issue 3: Model Files Not Found
**Solution:**
- Check if train_and_save_model.py ran successfully in build logs
- Verify Books_ratings.csv exists in repository
- Re-deploy the service

### Issue 4: Dataset Too Large for GitHub (>100MB)
**Solutions:**
1. **Use Git LFS:**
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   git add Books_ratings.csv
   git commit -m "Add dataset with LFS"
   git push
   ```

2. **Or Upload to Cloud Storage:**
   - Upload to Google Drive / Dropbox
   - Modify train_and_save_model.py to download from URL
   - Update build command

---

## Updating Your Deployment

### Method 1: Git Push (Automatic)
```bash
# Make changes to your files
git add .
git commit -m "Update: description of changes"
git push
```
Render automatically redeploys on push!

### Method 2: Manual Deploy (Render Dashboard)
1. Go to Render Dashboard
2. Select your service
3. Click **"Manual Deploy"** > **"Deploy latest commit"**

---

## Custom Domain (Optional)

### Free Custom Domain:
1. Go to Render Dashboard > Your Service
2. Click **"Settings"** tab
3. Scroll to **"Custom Domain"**
4. Add your domain
5. Update DNS records as instructed

---

## Monitoring

### View Logs:
1. Render Dashboard > Your Service
2. Click **"Logs"** tab
3. See real-time application logs

### Check Metrics:
1. Click **"Metrics"** tab
2. View CPU, memory usage
3. Monitor request counts

---

## Cost Estimate

### Free Tier (Current Setup):
- **Cost:** $0/month
- **Limitations:** 
  - Sleeps after 15 min inactivity
  - 750 hours/month
  - Shared resources

### Paid Tier (If Needed):
- **Starter:** $7/month
  - No sleep
  - Always-on
  - Better performance

---

## Share Your Project

### Your Live Demo URL:
```
https://book-review-analyzer.onrender.com
```

### GitHub Repository:
```
https://github.com/YOUR_USERNAME/amazon-book-review-analyzer
```

### Share on LinkedIn/Portfolio:
```
🎓 Final Project: Amazon Book Review Sentiment Analyzer
🔗 Live Demo: [Your Render URL]
📂 GitHub: [Your GitHub URL]
🎯 84.5% Accuracy using Logistic Regression
🛠️ Tech Stack: Python, Flask, scikit-learn, NLTK
```

---

## Success Checklist

- [ ] Repository pushed to GitHub
- [ ] Render service created
- [ ] Build completed successfully
- [ ] Application accessible at Render URL
- [ ] Can generate random reviews
- [ ] Sentiment analysis working
- [ ] About page displaying correctly
- [ ] Model info showing 84.5% accuracy
- [ ] No errors in Render logs

---

## Support

**Render Documentation:** https://render.com/docs
**Flask Documentation:** https://flask.palletsprojects.com/
**Deployment Issues:** Check Render community forum

---

**Congratulations! Your Amazon Book Review Analyzer is now live! 🎉**

**Team:** Maelyn L. Obejero and Angel Llatuna  
**Course:** IT 414 - Elective 3 Final Project
