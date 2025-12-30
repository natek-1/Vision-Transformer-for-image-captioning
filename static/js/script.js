document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-image');
    const generateBtn = document.getElementById('generate-btn');
    const resultSection = document.getElementById('result-section');
    const captionText = document.getElementById('caption-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const temperatureInput = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');

    let currentFile = null;

    // Drag & Drop
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

    function highlight(e) {
        dropArea.classList.add('active');
    }

    function unhighlight(e) {
        dropArea.classList.remove('active');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
                generateBtn.disabled = false;
                resultSection.classList.add('hidden');
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            previewContainer.classList.remove('hidden');
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent opening file dialog
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        generateBtn.disabled = true;
        resultSection.classList.add('hidden');
    });

    // Temperature slider
    temperatureInput.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });

    // Generate Caption
    generateBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        loadingOverlay.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('temperature', temperatureInput.value);

        try {
            const response = await fetch('/generate_caption', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate caption');
            }

            const data = await response.json();
            
            captionText.textContent = data.caption;
            resultSection.classList.remove('hidden');
            
            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('Error generating caption: ' + error.message);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    // Allow clicking on drop area to select file (if not clicking remove button)
    // Allow clicking on drop area to select file (if not clicking remove button or the label)
    dropArea.addEventListener('click', (e) => {
        // The label with 'for' attribute already triggers the input, so we avoid double-triggering
        if (e.target.closest('.file-upload-btn') || e.target ===  fileInput) {
            return;
        }
        
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });
});