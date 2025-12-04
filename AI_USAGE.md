# AI Tools Usage Documentation

**Student:** Roberto Berardi  
**Student Number:** 25419094  
**Project:** Dynamic Portfolio Clustering and Risk Profiling with Machine Learning  
**Course:** Advanced Programming - Fall 2025

---

## AI Tools Used

### Primary Tool: Claude (Anthropic)
- **Model:** Claude 3.5 Sonnet
- **Platform:** Claude.ai
- **Usage Period:** November-December 2025
- **Role:** Debugging assistant and coding mentor

---

## How I Used AI

I used Claude as a mentor to guide my learning and help when I got stuck, similar to working with a TA during office hours. Most of the code and all design decisions were mine, but Claude helped me debug issues and suggested improvements.

### My Work (~70%)

#### Project Design
- Defined the research question comparing clustering vs ML approaches
- Selected 50 U.S. stocks for analysis
- Designed three portfolio strategies with specific allocations
- Chose quarterly rebalancing and transaction cost parameters
- Decided on all evaluation metrics

#### Core Implementation
- Wrote data loading logic
- Implemented feature calculations (returns, volatility, Sharpe ratio, etc.)
- Set up clustering algorithms (K-means, GMM)
- Configured ML models (Ridge, Random Forest, XGBoost, Neural Network)
- Built portfolio construction and backtesting framework
- Created test scripts

#### Testing & Analysis
- Ran tests with different stock counts (5, 10, 50)
- Validated results against financial theory
- Interpreted why clustering outperformed ML
- Analyzed all performance metrics

---

## Where AI Helped (~30%)

### Debugging
Claude helped me diagnose and fix several bugs:

**Sharpe Ratio Calculation**
- I noticed results seemed wrong (values near zero)
- Claude helped identify the double calculation error
- I corrected the formula

**ML Training Data Issues**
- I encountered a missing column error
- Claude suggested restructuring the data flow
- I implemented the fix

**Cluster Encoding Problem**
- I got a string-to-float conversion error
- Claude reminded me about categorical encoding
- I added the numeric mapping

**Performance Optimization**
- I noticed slow execution times
- Claude suggested caching pre-calculated features
- I implemented the optimization

### Code Review
- Suggested using consistent random seeds for reproducibility
- Recommended modular file structure
- Advised on pandas best practices
- Helped with docstring formatting

### Documentation
- Helped structure README and PROPOSAL files
- Suggested what information to include
- Reviewed this AI usage document

---

## What I Learned

### Technical Skills
- Advanced pandas operations (rolling windows, data alignment)
- Scikit-learn model training and evaluation
- Code organization and modular design
- Debugging systematic approach
- Performance optimization techniques

### Financial Insights
- Why simpler methods can work better in noisy markets
- Impact of transaction costs on portfolio returns
- Importance of risk-adjusted metrics
- How adaptive rebalancing responds to market changes

---

## Understanding & Independence

**Can I explain the code?**  
Yes, I understand every function and the logic behind it. I know the mathematics (Sharpe ratio, PCA, K-means) and can explain the design choices.

**Could I work on this independently?**  
Yes, I can modify the code, add new features, or debug new issues. The patterns I learned apply to future projects.

**Is this my work?**  
Yes. The research design, implementation logic, and results interpretation are mine. Claude helped with debugging and best practices, similar to a TA or mentor.

---

## Comparison to Traditional Help

Using Claude was comparable to:
- Asking questions during office hours
- Getting code review from a classmate
- Looking up solutions on Stack Overflow
- Working with a study group

The main difference: Claude provided immediate, 24/7 feedback.

---

## Declaration

I, Roberto Berardi (Student #25419094), used Claude as a learning tool and debugging assistant. This project represents my own work and understanding. I can explain all code submitted and could recreate similar projects independently.

---

**Roberto Berardi**  
December 4, 2025
