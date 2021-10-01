
$(document).ready(function(){

  $('html,body').animate({
        scrollTop: $('#main-container').offset().top-100
    }, 'slow');
  $('.circle-loader').toggleClass('load-complete');
  $('.checkmark').toggle();
   $(window).scroll( function(){
    $('.fade-scroll').each( function(i){
            
            var bottom_of_object = $(this).offset().top + $(this).outerHeight();
            var bottom_of_window = $(window).scrollTop() + $(window).height();
            if( bottom_of_window > bottom_of_object ){
               
                  $(this).removeClass("fade-scroll");
                  $(this).addClass( "left-right-fade" );
               
            }
            
        }); 
    
    });
   var s="";
   specs.forEach(function(spec){
    s="<div class='col-4 col-md-3 p-2'><a href='./doctors.html'>      <div   class='col-12 shadow rounded text-center p-2 specialist-item'>         <img class='img-fluid' width='100' src='./assets/doctors/{spec}.svg'>         <br> <p class='raleway text-capitalize font-weight-bold overflow-wrap'>{spec}</p>      </div></a></div>"
   });
   $('.specialist-list').html();
  });
$("#doctorsearch").on("keyup", function() {
    var value = $(this).val().toLowerCase();
    $(".doctor-list").filter(function() {
      $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
    });
  });
$('#isPavail').change(function() {
  var isPavail=document.getElementById('isPavail');
  if(isPavail.checked==true){
  $('.hospital-detail').toggleClass('d-none');
  }
  else
  {
    $('.hospital-detail').toggleClass('d-none');

  }  
});
	$('#asPatient').click(function () {
		$('#roleModal').modal("hide");
		$('#patientModal').modal("show");
    });
  $('#asDoctor').click(function () {
    $('#roleModal').modal("hide");
    $('#doctorModal').modal("show");
    });
$(function() {
    $(document).on("change",".uploadFile", function()
    {
        var uploadFile = $(this);
        var files = !!this.files ? this.files : [];
        if (!files.length || !window.FileReader) return;
 
        if (/^image/.test( files[0].type)){
            var reader = new FileReader(); 
            reader.readAsDataURL(files[0]);
 
            reader.onloadend = function(){ 
uploadFile.closest(".imgUp").find('.imagePreview').css("background-image", "url("+this.result+")");
            }
        }
      
    });
});

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    
    reader.onload = function(e) {
      $('#img-prev').attr('src', e.target.result);
      $('#img-prev').removeClass("p-5");
      $('#img-prev').addClass("p-1");
    }
    
    reader.readAsDataURL(input.files[0]); // convert to base64 string
  }
}

$("#doctor-img").change(function() {
  readURL(this);
});

function readFile(input) {
  $(".dropzone-wrapper").addClass('d-none');
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function(e) {
      var htmlPreview =
        '<img class="img-fluid" src="' + e.target.result + '" />' +
        '<p>' + input.files[0].name + '</p>';
      var wrapperZone = $(input).parent();
      var previewZone = $(input).parent().parent().find('.preview-zone');
      var boxZone = $(input).parent().parent().find('.preview-zone').find('.box').find('.box-body');

      wrapperZone.removeClass('dragover');
      $('.remove-preview').removeClass('d-none');
      previewZone.removeClass('hidden');
      boxZone.empty();
      boxZone.removeClass('d-none');
      boxZone.append(htmlPreview);
    };

    reader.readAsDataURL(input.files[0]);
  }
}

function reset(e) {
  $('.remove-preview').addClass('d-none');
  e.wrap('<form>').closest('form').get(0).reset();
  e.unwrap();
}

$(".dropzone").change(function() {
  readFile(this);
});

$('.dropzone-wrapper').on('dragover', function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).addClass('dragover');
});

$('.dropzone-wrapper').on('dragleave', function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).removeClass('dragover');
});

$('.remove-preview').on('click', function() {
  var boxZone = $(this).parents('.preview-zone').find('.box-body');
  var dropzone = $(this).parents('.form-group').find('.dropzone');
  boxZone.empty();
  $('.box-body').addClass('d-none');
  $(".dropzone-wrapper").removeClass('d-none');
  reset(dropzone);
});
