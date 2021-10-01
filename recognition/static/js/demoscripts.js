$('#filtertype').change(function(){
var ft=$(this).val()
  if(ft=="date"){
     $('.filterbydate').toggleClass("d-none");
     $('.filterbyemp').addClass("d-none");

  }
  else if(ft=="emp"){
    $('.filterbyemp').toggleClass("d-none");
$('.filterbydate').addClass("d-none"); 
  
  }
});
$('#filterbydate-btn').click(function(){
    var date=$('#filterbydate-date').val()  
  alert(date);
});
$('#filterbyemp-btn').click(function(){
    var name=$('#filterbyemp-name').val()
    var date1=$('#filterbyemp-date1').val()
    var date2=$('#filterbyemp-date2').val()  
  alert(name+date1+date2);
});
// $('.alog').click(function() {
//   var adate=$(this).attr('adate');
//   var url="/view_attendance_date/"+adate;
//   alert(adate);
// });
  $('#filterData').click(function(){
    var adate=$('#ondate').val();
     var sql="";
    if(adate!=""){
  sql="SELECT * FROM records WHERE  date= '"+adate+"'' LIMIT 50";
      }
     alert(sql);
  });


  //clock
  let clock = () => {
  let date = new Date();
  let hrs = date.getHours();
  let mins = date.getMinutes();
  let secs = date.getSeconds();
  let period = "AM";
  if (hrs == 0) {
    hrs = 12;
  } else if (hrs >= 12) {
    hrs = hrs - 12;
    period = "PM";
  }
  hrs = hrs < 10 ? "0" + hrs : hrs;
  mins = mins < 10 ? "0" + mins : mins;
  secs = secs < 10 ? "0" + secs : secs;

  let time = `${hrs}:${mins}:${secs}:${period}`;
  document.getElementById("clock").innerText = time;
  setTimeout(clock, 1000);
};

clock();