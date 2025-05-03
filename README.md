# Image Captioning with Vision Transformers

## Overview
This project implements an **image captioning model** using **Vision Transformers (ViTs)**. The model is initially trained on the **COCO dataset**, with plans to expand to **Visual Genome** and **LAION-5B** for improved generalization. The application is deployed using **Docker, S3, and EC2 on AWS**, with CI/CD automation through **GitHub Actions**.

## Features
- **Image Captioning**: Generate natural language descriptions for images.
- **Vision Transformers (ViTs)**: Use transformer-based architectures for feature extraction.
- **Datasets**:
  - COCO (initial dataset)
  - Visual Genome (future expansion)
  - LAION-5B (for large-scale learning)
- **Deployment**:
  - Containerized with **Docker**
  - Cloud storage via **AWS S3**
  - Deployed on **AWS EC2**
  - Automated CI/CD using **GitHub Actions**

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Docker
- AWS CLI configured
- GitHub Actions enabled

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/natek-1/Vision-Transformer-for-image-captioning.git
   cd Vision-Transformer-for-image-captioning
   ```
2. **Install dependencies**:
   ```bash
   conda create -n image_captioning python=3.10 -y
   conda activate image_captioning
   pip install -r requirements.txt
   ```
3. **Set up AWS credentials**:
   ```bash
   aws configure
   ```
4. **Run the application locally**:
   ```bash
   python app.py
   ```

## Training the Model
1. **Prepare the COCO dataset**:
   - Download from [COCO Dataset](https://cocodataset.org/#download)
   - Preprocess images and annotations
2. **Train the model**:
   ```bash
   python train.py --dataset coco --epochs 10
   ```
3. **Evaluate the model**:
   ```bash
   python evaluate.py --dataset coco
   ```

## Deployment
### Docker Containerization
1. **Build the Docker image**:
   ```bash
   docker build -t image-captioning .
   ```
2. **Run the container locally**:
   ```bash
   docker run -p 5000:5000 image-captioning
   ```

### AWS Deployment
1. **Push Docker image to AWS ECR**:
   ```bash
   aws ecr create-repository --repository-name image-captioning
   docker tag image-captioning:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/image-captioning:latest
   docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/image-captioning:latest
   ```
2. **Deploy on EC2**:
   - Launch an EC2 instance
   - Pull the Docker image
   - Run the container

### CI/CD with GitHub Actions
- Automated training and deployment workflow using GitHub Actions.
- Workflow file: `.github/workflows/deploy.yml`

## Usage
- **Upload an image**
- **Get a caption prediction**

Example API request:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## Reference

This project is based on the following paper:

**S. Mishra et al.**, *Image Caption Generation using Vision Transformer and GPT Architecture*,  
2024 2nd International Conference on Advancement in Computation & Computer Technologies (InCACCT), Gharuan, India, 2024, pp. 1–6.  
[DOI: 10.1109/InCACCT61598.2024.10551257](https://doi.org/10.1109/InCACCT61598.2024.10551257)

```bibtex
@inproceedings{mishra2024image,
  author    = {S. Mishra and others},
  title     = {Image Caption Generation using Vision Transformer and GPT Architecture},
  booktitle = {2024 2nd International Conference on Advancement in Computation & Computer Technologies (InCACCT)},
  year      = {2024},
  pages     = {1--6},
  doi       = {10.1109/InCACCT61598.2024.10551257},
  keywords  = {Image Captioning, Vision Transformer, GPT-2, Encoder-Decoder}
}

## Future Work
- Support for **Visual Genome** and **LAION-5B** datasets.
- Fine-tuning with **self-supervised learning**.
- Integrate **LLMs for text refinement**.

## License
MIT License

---
**Author**: Nathan Gabriel Kuissi

