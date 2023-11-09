# Use an official Python image as base
FROM python:3.8

# Set the working directory to /usr/src/app
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure wait-for-it.sh is executable
RUN chmod +x wait-for-it.sh

