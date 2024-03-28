$(document).ready(function() {
  var header = $(".header");
  var footer = $(".footer");

  // Array di colori consentiti per lo sfondo, scelti manualmente per contrastare bene con il testo bianco
  var allowedColors = ["#808080", "#333333", "#483C32", "#708090", "#C0C0C0", "#36454F", "#B2BEB5", "#800080", "#0000FF", "#008000"
];

  // Genera un colore casuale consentito per lo sfondo
  function randomColor() {
      return allowedColors[Math.floor(Math.random() * allowedColors.length)];
  }

  // Imposta il colore casuale per l'area di intestazione e il piè di pagina
  var backgroundColor = randomColor();
  header.css("background-color", backgroundColor);
  footer.css("background-color", backgroundColor);

  // Imposta il colore del testo su bianco per garantire un buon contrasto
  header.css("color", "#FFFFFF");
  footer.css("color", "#FFFFFF");
});

function saveImage() {
    $("#train-model-button").hide();
    fetch("/save_image", { method: "POST" })
        .then(response => {
            if (response.ok) {
                $("#success-message").html("<p>Image saved with success!</p>");
            } else {
                $("#success-message").html("<p>Failed to save image</p>");
            }
        })
        .catch(error => {
            console.error('Error saving image:', error);
            $("#success-message").html("<p>Failed to save image</p>");
        });
}

function uploadImage(event) {
  // Allows to upload an image and update the content of the div with the uploaded image
  event.preventDefault();
  var formData = new FormData();
  formData.append('file', document.getElementById('image-selector').files[0]);

  fetch('/upload', {
      method: 'POST',
      body: formData
  })
  .then(response => response.text())
  .then(data => {
      // Aggiorna il contenuto del div con l'immagine caricata
      document.getElementById('uploaded-image').innerHTML = `<img src="${data}" alt="uploaded image" style="max-width: 100%; max-height: 500px;">`;
  })
  .catch(error => {
      console.error('Error:', error);
  });
}

document.getElementById('upload-form').addEventListener('submit', uploadImage);

function trainModel() {
    $("#success-message").empty();
    $("#train-model-button").hide();
    $("#loadingModal").modal("show");
    // Mostra la barra di avanzamento e il messaggio
    $("#progress-container").show();
    $(".progress-bar").css("width", "0%");
    $("#progress-message").text("Training in progress...");

    // Scorri la finestra fino alla fine della pagina
    $('html, body').animate({
      scrollTop: $(document).height() - $(window).height()
    }, 1000);
  
    let progress = 0;
    const interval = setInterval(function () {
      progress += 10;
      $(".progress-bar").css("width", progress + "%");
      $(".progress-bar").attr("aria-valuenow", progress);
      $("#progress-message").text("Training in progress... " + progress + "%");
      if (progress > 100) {
        progress = 0;
        $(".progress-bar").css("width", progress + "%");
        $("#progress-message").text("Training in progress... " + progress + "%");
      }
    }, 1000);
  
    $.post("/train_model", function (data) {
      $("#results").html(data);
      $("#loadingModal").modal("hide");
      $("#train-model-button").show();
      // Nascondi la barra di avanzamento dopo il completamento del training
      $("#progress-container").hide();
      clearInterval(interval);

      // Scorri la finestra fino alla fine della pagina
      $('html, body').animate({
        scrollTop: $(document).height() - $(window).height()
      }, 1000);
    });
  }

  function processImage() {
    $("#success-message").empty();
    $("#process-image-button").hide();
    $("#loadingModal").modal("show");
    // Mostra la barra di avanzamento e il messaggio
    $("#progress-container").show();
    $(".progress-bar").css("width", "0%");
    $("#progress-message").text("Processing image...");

    // Scorri la finestra fino alla fine della pagina
    $('html, body').animate({
      scrollTop: $(document).height() - $(window).height()
    }, 1000);
  
    let progress = 0;
    const interval = setInterval(function () {
      progress += 25;
      $(".progress-bar").css("width", progress + "%");
      $(".progress-bar").attr("aria-valuenow", progress);
      $("#progress-message").text("Processing image... " + progress + "%");
      if (progress > 100) {
        progress = 0;
        $(".progress-bar").css("width", progress + "%");
        $("#progress-message").text("Processing image... " + progress + "%");
      }
    }, 1000);
  
    $.post("/process_image", function (data) {
      $("#results").html(data);
      $("#loadingModal").modal("hide");
      $("#process-image-button").show();
      // Nascondi la barra di avanzamento dopo il completamento dell'elaborazione dell'immagine
      $("#progress-container").hide();
      clearInterval(interval);

      // Scorri la finestra fino alla fine della pagina
      $('html, body').animate({
        scrollTop: $(document).height() - $(window).height()
      }, 1000);
    });
  }
