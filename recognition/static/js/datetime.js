
var defaults = {
  calendarWeeks: true,
  showClear: true,
  showClose: true,
  allowInputToggle: true,
  useCurrent: false,
  maxDate:new Date(),
  ignoreReadonly: true,
  toolbarPlacement: 'top',
  locale: 'en',
  icons: {
    time: 'fa fa-clock-o',
    date: 'fa fa-calendar',
    up: 'fa fa-angle-up',
    down: 'fa fa-angle-down',
    previous: 'fa fa-angle-left',
    next: 'fa fa-angle-right',
    today: 'fa fa-dot-circle-o',
    clear: 'fa fa-trash',
    close: 'fa fa-times'
  }
};


$(function() {
  var optionsDate = $.extend({}, defaults, {format:'YYYY-MM-DD'});
  var optionsTime = $.extend({}, defaults, {format:'HH:mm'});
  
  $('.datepicker').datetimepicker(optionsDate);
  $('.timepicker1').datetimepicker(optionsTime);
  $('.timepicker2').datetimepicker(optionsTime);
  // $('.datetimepicker').datetimepicker(optionsDatetime).on('dp.change',  function () {
  //           alert($('.datepicker input').val());
  //         });

});
