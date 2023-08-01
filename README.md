# Machine Learning Ops Challenge

This project involved deploying an image segmentation model and setting up CI/CD pipelines.

## Challenges

**Model Deployment**

- Serving segmentation model with FastAPI
- Handling image file uploads and downloads
- Switching to in-memory streams to avoid filesystem issues

**Testing**

- Writing pytest unit tests for core model and API  
- Testing locally and on GitHub Actions

**CI/CD Pipelines**

- Creating GitHub Actions workflows
- Deploying API to Render platform


## Solutions

- Implemented FastAPI endpoint for image segmentation
- Dockerized API for container deployments
- Utilized in-memory streams to remove filesystem dependencies
- Wrote pytest tests for model and API validation
- Configured GitHub Actions to run tests on push

## TODO

- [x] Create one or two minimal tests
  - [x] Test locally
- [x] Create a deployment on a realtime endpoint
  - [x] Deploy to cloud 
  - [ ] Test remotely (Postman)
- [ ] Create a minimal CI pipeline with GitHub Actions
  - [ ] Test remotely (GitHub Actions) 
- [ ] Create a minimal CD pipeline
  - [ ] Complete CI/CD chain

Let me know if you would like me to expand or modify any part of this README!