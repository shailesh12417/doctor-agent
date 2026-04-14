# 1. PROVIDER CONFIGURATION
provider "aws" {
  region = "eu-north-1" # Updated to Stockholm
}

# 2. DATA SOURCE TO FIND THE LATEST UBUNTU 24.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# 3. SECURITY GROUP (Allows SSH and Web Traffic)
resource "aws_security_group" "app_sg" {
  name        = "doctor-agent-sg"
  description = "Allow SSH and HTTP"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 4. EC2 INSTANCE
resource "aws_instance" "app_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.micro" 
  key_name               = "terraform" # Ensure this key exists in Stockholm!
  vpc_security_group_ids = [aws_security_group.app_sg.id]

  # Provisioning script to install Docker
  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y docker.io
              sudo systemctl start docker
              sudo systemctl enable docker
              EOF

  tags = {
    Name = "DoctorAgentServer"
  }
}

# 5. OUTPUTS (Prints the IP to your terminal)
output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}
