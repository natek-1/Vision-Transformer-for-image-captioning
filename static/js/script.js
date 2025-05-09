document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileUpload = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image');
    const generateBtn = document.getElementById('generate-btn');
    const resultSection = document.getElementById('result-section');
    const resultImage = document.getElementById('result-image');
    const captionText = document.getElementById('caption-text');
    const loadingSpinner = document.getElementById('loading');
    
    // Parameter elements
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const topPSlider = document.getElementById('top-p');
    const topPValue = document.getElementById('top-p-value');
    const topKSlider = document.getElementById('top-k');
    const topKValue = document.getElementById('top-k-value');
    
    // Result parameter display elements
    const usedTemperature = document.getElementById('used-temperature');
    const usedTopP = document.getElementById('used-top-p');
    const usedTopK = document.getElementById('used-top-k');
    
    // Selected file
    let selectedFile = null;
    
    // Event Listeners for file uploading
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle file drop
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFiles(files);
        }
    }
    
    // Handle file selection via button
    fileUpload.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(this.files);
        }
    });
    
    // Process selected files
    function handleFiles(files) {
        const file = files[0];
        
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        
        selectedFile = file;
        displayImagePreview(file);
        generateBtn.disabled = false;
    }
    
    // Display image preview
    function displayImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            document.querySelector('.drop-message').classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        
        reader.readAsDataURL(file);
    }
    
    // Remove selected image
    removeImageBtn.addEventListener('click', function() {
        previewContainer.classList.add('hidden');
        document.querySelector('.drop-message').classList.remove('hidden');
        fileUpload.value = '';
        selectedFile = null;
        generateBtn.disabled = true;
    });
    
    // Update parameter value displays
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });
    
    topPSlider.addEventListener('input', function() {
        topPValue.textContent = this.value;
    });
    
    topKSlider.addEventListener('input', function() {
        topKValue.textContent = this.value;
    });
    
    // Generate caption
    generateBtn.addEventListener('click', function() {
        if (!selectedFile) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.classList.remove('hidden');
        resultSection.classList.add('hidden');
        
        // Get parameter values
        const temperature = temperatureSlider.value;
        const topP = topPSlider.value;
        const topK = topKSlider.value;
        
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('temperature', temperature);
        formData.append('top_p', topP);
        formData.append('top_k', topK);
        
        // Send request to server
        fetch('/generate_caption', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loadingSpinner.classList.add('hidden');
            
            // Display results
            resultImage.src = `data:image/jpeg;base64,${data.image}`;
            captionText.textContent = data.caption;
            
            // Display used parameters
            usedTemperature.textContent = data.parameters.temperature;
            usedTopP.textContent = data.parameters.top_p;
            usedTopK.textContent = data.parameters.top_k;
            
            // Show result section
            resultSection.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            loadingSpinner.classList.add('hidden');
            alert('An error occurred while generating the caption. Please try again.');
        });
    });
    
    // Click on drop area to trigger file upload
    dropArea.addEventListener('click', function() {
        fileUpload.click();
    });
});