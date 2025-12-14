# 📋 Deployment Checklist

## Pre-Deployment (Local Testing)

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files generated (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`)
- [ ] Flask server runs locally (`python app.py`)
- [ ] Frontend accessible at `http://localhost:5000`
- [ ] "Generate Review" button works
- [ ] "Analyze Sentiment" button works
- [ ] About page displays correctly
- [ ] No console errors in browser

## Repository Setup

- [ ] Git initialized (`git init`)
- [ ] All files added (`git add .`)
- [ ] Initial commit created (`git commit -m "Initial commit"`)
- [ ] GitHub repository created
- [ ] Repository is PUBLIC (required for free Render)
- [ ] Code pushed to GitHub (`git push origin main`)
- [ ] Books_ratings.csv uploaded (check file size <100MB)

## Files to Include

Essential files for deployment:
- [ ] `app.py` (Flask backend)
- [ ] `index.html` (Frontend)
- [ ] `train_and_save_model.py` (Model training)
- [ ] `Books_ratings.csv` (Dataset)
- [ ] `requirements.txt` (Dependencies)
- [ ] `render.yaml` (Render config)
- [ ] `.gitignore` (Git ignore)
- [ ] `README.md` (Documentation)
- [ ] `DEPLOYMENT_GUIDE.md` (Instructions)

Optional but recommended:
- [ ] `book_review_dataset.ipynb` (Jupyter notebook)
- [ ] `section-Obejero-Llatuna.md` (Full documentation)
- [ ] `test_api.py` (API testing)
- [ ] `start.bat` (Quick start script)

## Render Configuration

- [ ] Render account created
- [ ] GitHub authorized in Render
- [ ] Repository connected
- [ ] Service type: **Web Service**
- [ ] Runtime: **Python 3**
- [ ] Build command: `pip install -r requirements.txt && python train_and_save_model.py`
- [ ] Start command: `gunicorn app:app`
- [ ] Instance type: **Free**
- [ ] Region selected (e.g., Oregon)
- [ ] Auto-deploy enabled (optional)

## Environment Variables (Optional)

- [ ] `PYTHON_VERSION` = `3.11.0`
- [ ] `NLTK_DATA` = `/opt/render/project/src/nltk_data`

## Deployment Process

- [ ] "Create Web Service" clicked
- [ ] Build started (check logs)
- [ ] Dependencies installed successfully
- [ ] NLTK data downloaded
- [ ] Model training completed (3-5 min)
- [ ] Build succeeded (green checkmark)
- [ ] Service deployed
- [ ] URL generated (e.g., `https://your-app.onrender.com`)

## Post-Deployment Testing

- [ ] Homepage loads (`https://your-app.onrender.com`)
- [ ] No 404 errors
- [ ] Frontend displays correctly
- [ ] "Generate Review" fetches real reviews
- [ ] "Analyze Sentiment" returns predictions
- [ ] Confidence scores showing
- [ ] Star ratings displaying
- [ ] About page accessible
- [ ] Model info table visible
- [ ] Dataset description showing

## API Testing

Test using curl or Postman:

- [ ] **Predict Sentiment Endpoint:**
  ```bash
  curl -X POST https://your-app.onrender.com/predict_sentiment \
    -H "Content-Type: application/json" \
    -d '{"text": "This book was amazing!"}'
  ```
  Expected: JSON with sentiment, confidence

- [ ] **Random Review Endpoint:**
  ```bash
  curl https://your-app.onrender.com/get_random_review
  ```
  Expected: JSON with review data

- [ ] **Homepage:**
  ```bash
  curl https://your-app.onrender.com
  ```
  Expected: HTML content

## Performance Checks

- [ ] First load time < 60 seconds (cold start)
- [ ] Subsequent requests < 2 seconds
- [ ] Model predictions < 500ms
- [ ] No memory errors in logs
- [ ] No timeout errors

## Documentation

- [ ] README.md updated with live URL
- [ ] DEPLOYMENT_GUIDE.md reviewed
- [ ] Screenshots taken (optional)
- [ ] Demo video created (optional)

## Sharing

- [ ] Live URL tested in incognito/private mode
- [ ] URL shared with team members
- [ ] Project added to portfolio
- [ ] LinkedIn post created (optional)
- [ ] GitHub repository description updated

## Monitoring

- [ ] Render dashboard accessible
- [ ] Logs viewable
- [ ] Metrics showing
- [ ] Email notifications configured (optional)

## Troubleshooting Checklist

If deployment fails:

- [ ] Check build logs in Render
- [ ] Verify all files in GitHub
- [ ] Check requirements.txt syntax
- [ ] Ensure Books_ratings.csv uploaded
- [ ] Test locally first
- [ ] Review error messages
- [ ] Check Render status page
- [ ] Consult DEPLOYMENT_GUIDE.md

## Final Verification

- [ ] Application URL: `_________________________`
- [ ] GitHub Repository: `_________________________`
- [ ] Deployment Date: `_________________________`
- [ ] Team Members Notified: ☐ Yes ☐ No
- [ ] Documentation Complete: ☐ Yes ☐ No
- [ ] Ready for Demo: ☐ Yes ☐ No

---

## Success Criteria

✅ All checkboxes above are checked  
✅ Application accessible via public URL  
✅ All features working correctly  
✅ No errors in production logs  
✅ Documentation complete and accurate  

---

**Deployment Completed By:** _________________________  
**Date:** _________________________  
**Render URL:** _________________________  
**Status:** ☐ Success ☐ Issues (see notes)

**Notes:**
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

---

**🎉 Congratulations on deploying your Amazon Book Review Analyzer!**
