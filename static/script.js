const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const evaluateBtn = document.getElementById('evaluate-btn');
const resultText = document.getElementById('result')
let uploadedFileName = '';

uploadBox.addEventListener('click', () => fileInput.click());

evaluateBtn.addEventListener('click', () => resultText.style.display = 'block')

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        previewImage(file);
        uploadFile(file);
    }
});

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file && file.type.startsWith('image/')) {
        previewImage(file);
        uploadFile(file);
    }
});

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        uploadedFileName = "meme.png";
        evaluateBtn.style.display = 'block';
        if (resultText != null) resultText.style.display = 'None';
    })
    .catch((reason) => alert(reason));
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadBox.style.backgroundImage = `url(${e.target.result})`;
        uploadBox.style.backgroundSize = 'cover';
        uploadBox.style.backgroundPosition = 'center';
        uploadBox.textContent = ''; // Clear text inside the upload box
    };
    reader.readAsDataURL(file);
}
