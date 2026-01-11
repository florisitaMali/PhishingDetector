Phishing Dataset for Machine Learning

Link: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
Source:https://data.mendeley.com/datasets/h3cgnj8hft/1

-----------------------------
Dataset Overview
-----------------------------

This dataset is designed for phishing website detection and contains structured features extracted from website URLs and webpage content. Each instance represents a single website and is labeled as either phishing or legitimate. The dataset is suitable for supervised machine learning classification tasks, particularly in cybersecurity applications.

- Dataset type: Binary classification
- Domain: Cybersecurity / Phishing detection
- Data format: CSV
- Target variable: CLASS_LABEL
- Size: 10,043 instances
- Features: 42 input features + 1 class label
- Feature Type: 
    *   Numeric (e.g., URL length, number of dots, number of dashes)
    *   Binary / discrete indicators (e.g., presence of iframe, missing title)
    *   Target: Binary classification (Phishing vs Legitimate)

-------------------------------
Target Variable
-------------------------------
CLASS_LABEL
- 1 → Phishing website
- 0 → Legitimate website

-------------------------------
Feature Description
-------------------------------
The dataset consists entirely of structured numerical and binary features, making it directly compatible with tree-based machine learning models such as Decision Trees and Random Forests.

URL-BASED FEATURES

These features describe the structure and composition of the URL:
- NumDots: Number of dots in the URL
- SubdomainLevel: Depth of subdomains
- PathLevel: Number of path segments
- UrlLength: Total URL length
- NumDash, NumUnderscore, NumPercent, NumHash,NumAmpersand: Counts of special characters
- NumNumericChars: Number of numeric characters
- AtSymbol, TildeSymbol, DoubleSlashInPath:Presence of suspicious URL symbol
- NoHttps: Indicates absence of HTTPS
- IpAddress: Indicates use of an IP address instead of a domain name

DOMAIN AND HOSTNAME FEATURES

These features capture domain-related anomalies:
- HostnameLength: Length of the hostname
- DomainInSubdomains: Domain name appears in subdomains
- DomainInPaths: Domain name appears in URL path
- HttpsInHostname: HTTPS token appears in hostname
- FrequentDomainNameMismatch: Mismatch between domain and content

CONTENT AND HTML-BASED FEATURES

These features are extracted from webpage content and HTML structure:
- PctExtHyperlinks: Percentage of external hyperlinks
- PctExtResourceUrls: Percentage of external resource URLs
- ExtFavicon: External favicon usage
- IframeOrFrame: Presence of iframe or frame tags
- MissingTitle: Missing HTML title tag
- ImagesOnlyInForm: Forms containing only images
- ExtMetaScriptLinkRT: External metadata or scripts

FORM AND INTERACTION FEATURES

These features detect suspicious user interaction behaviors:
- InsecureForms: Forms submitted over insecure channels
- RelativeFormAction, ExtFormAction, AbnormalFormAction: Form action anomalies
- SubmitInfoToEmail: Data submission via email
- RightClickDisabled: Right-click disabled
- PopUpWindow: Presence of pop-up windows

-----------------------------------
Acknowledgements
-----------------------------------

Tan, Choon Lin (2018), “Phishing Dataset for Machine Learning: Feature Evaluation”, Mendeley Data, V1, doi: 10.17632/h3cgnj8hft.1
Source of the Dataset.
