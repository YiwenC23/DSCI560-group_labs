document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const pdfFilesInput = document.getElementById('pdfFiles');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileList = document.getElementById('fileList');
    const analyzeButton = document.getElementById('analyzeButton');
    const analyzeStatus = document.getElementById('analyzeStatus');
    
    let uploadedFiles = [];
    
    // Handle file selection UI update
    pdfFilesInput.addEventListener('change', function(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            let fileNames = files.map(file => `<li>${file.name}</li>`).join('');
            fileList.innerHTML = `<p>Selected files:</p><ul>${fileNames}</ul>`;
        } else {
            fileList.innerHTML = '';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const files = pdfFilesInput.files;
        if (files.length === 0) {
            uploadStatus.innerHTML = '<p style="color: red;">Please select at least one PDF file</p>';
            return;
        }
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('pdfFiles', files[i]);
        }
        
        uploadStatus.innerHTML = '<p>Uploading files...</p>';
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFiles = data.files;
                uploadStatus.innerHTML = `<p style="color: green;">Successfully uploaded ${uploadedFiles.length} file(s)</p>`;
                analyzeButton.disabled = false;
            } else {
                uploadStatus.innerHTML = `<p style="color: red;">${data.error}</p>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.innerHTML = '<p style="color: red;">Error uploading files</p>';
        });
    });
    
    // Handle analyze button click
    analyzeButton.addEventListener('click', function() {
        if (uploadedFiles.length === 0) {
            analyzeStatus.innerHTML = '<p style="color: red;">No files to analyze</p>';
            return;
        }
        
        analyzeStatus.innerHTML = '<p>Analyzing files...</p>';
        analyzeButton.disabled = true;
        
        // Redirect to the analyze page
        window.location.href = '/analyze';
    });
});