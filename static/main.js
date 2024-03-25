function saveImage() {
    $("#train-model-button").hide();
    fetch("/save_image", { method: "POST" });
}

function trainModel() {
    $("#train-model-button").hide();
    $.post("/train_model", function (data) {
        $("#results").html(data);
    });
}

function processImage() {
    $("#train-model-button").hide();
    $.post("/process_image", function (data) {
      $("#results").html(data);
    });
}

