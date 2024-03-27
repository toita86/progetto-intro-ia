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

