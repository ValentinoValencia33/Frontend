const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const emotionText = document.getElementById('emotion');
const confidenceText = document.getElementById('confidence');
const fileInput = document.getElementById('fileInput');

let processing = false;

// Acceder a la cámara
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error("Error al acceder a la cámara: ", err));

setInterval(async () => {
    if (processing) return;
    processing = true;

    // Ajustar tamaño del canvas al video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Dibujar el frame del video en el canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convertir canvas a blob y enviarlo al backend
    canvas.toBlob(async (blob) => {
        if (!blob) {
            console.error("Error: No se pudo generar el blob de la imagen.");
            processing = false;
            return;
        }

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const response = await fetch("http://localhost:8000/predict/", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            console.log("Respuesta del backend:", data); 

            if (data.emotion) {
                emotionText.innerText = data.emotion;
                confidenceText.innerText = (data.confidence * 100).toFixed(2) + "%";

                // Dibujar rectángulo en los rostros detectados
                if (data.faces && data.faces.length > 0) {
                    ctx.strokeStyle = "#00FF00"; 
                    ctx.lineWidth = 3;

                    data.faces.forEach(face => {
                        const [x, y, w, h] = face;
                        ctx.strokeRect(x, y, w, h);
                    });
                }
            }
        } catch (error) {
            console.error("Error en la predicción:", error);
        }

        processing = false;
    }, "image/jpeg");
}, 2000);

// Evento para procesar imágenes cargadas
fileInput.addEventListener("change", async function() {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8000/predict/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log("Respuesta del backend (imagen):", data);

        if (data.emotion) {
            emotionText.innerText = data.emotion;
            confidenceText.innerText = (data.confidence * 100).toFixed(2) + "%";
        }
    } catch (error) {
        console.error("Error al procesar la imagen:", error);
    }
});
