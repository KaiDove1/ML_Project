# Selective Pesticide & Fertilizer Application for Stormwater Runoff Mitigation

## Project Overview
Stormwater runoff in Virginia, primarily from excess pesticide and fertilizer application, poses significant threats to local ecosystems. This project aims to reduce runoff by creating a machine learning (ML) solution that selectively applies pesticides and fertilizers only when necessary. Using image recognition, this solution identifies plant health and disables application if treatment is unnecessary, optimizing resources and minimizing environmental impact.

## Data Sources
- **Chesapeake Bay Program**: [Agricultural Runoff Data](https://www.chesapeakebay.net/issues/threats-to-the-bay/agricultural-runoff)
- **Satellite Image Time Series Dataset**: [GitHub Repository](https://github.com/corentin-dfg/Satellite-Image-Time-Series-Datasets?tab=readme-ov-file)
- **USDA NASS Data and Statistics**: [USDA NASS](https://www.nass.usda.gov/Data_and_Statistics/)

## Setting Up Project

### Installing Packages

All packages are in the `requirements.txt` file in the root of this project. You can install all of them at once using the following command:
```bash
pip install -r requirements.txt
```

### Authenticating with Earth Engine

1. Ask Michael for the Service Account JSON file and place it in the root directory of the project.
2. Create a `.env` file with the following contents:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account-json>.json
```
3. Run `source .env` to load the environment variables. In the future, you can place any environment variables in here in the same format and run `source .env` to load them.
4. Install the `gcloud` CLI. On Mac, you can do this with `brew install --cask google-cloud-sdk`.
5. Run `gcloud config set project ee-myfatemi04`.
6. Run `gcloud auth login`.
7. Run `earthengine authenticate`.
