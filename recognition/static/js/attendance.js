$(document).ready(function(){
    contact_poll_id = setInterval("contact_poll()",5000);
});

function contact_poll() {
    $.ajax({
        url: 'update_attendance',
        type: 'POST',
        datatype: 'text',
        success: function(data) {
            var contacts_html = "";
            if( data!= "" ) {
            
                $("#contacts").empty();
                $("#contacts").append(contacts_html);
            }
        },
        complete:function(data) {
            //setTimeout(contact_poll,5000);
        }
    });
}