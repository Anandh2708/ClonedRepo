

function changeTable(){
    var sec = document.getElementById("accuracyType");
    var selected = sec.options[sec.selectedIndex].value;

     if (selected == "maxAcc") {

         document.getElementById("maxAccTable").style.display = "block";
         document.getElementById("oneWeekAccTable").style.display = "none";
         document.getElementById("oneMonthAccTable").style.display = "none";
         document.getElementById("sixMonthAccTable").style.display = "none";
         document.getElementById("oneYearAccTable").style.display = "none";
     }else if(selected == "oneWeekAcc") {

         document.getElementById("maxAccTable").style.display = "none";
         document.getElementById("oneWeekAccTable").style.display = "block";
         document.getElementById("oneMonthAccTable").style.display = "none";
         document.getElementById("sixMonthAccTable").style.display = "none";
         document.getElementById("oneYearAccTable").style.display = "none";
     }else if(selected == "oneMonthAcc") {

         document.getElementById("maxAccTable").style.display = "none";
         document.getElementById("oneWeekAccTable").style.display = "none";
         document.getElementById("oneMonthAccTable").style.display = "block";
         document.getElementById("sixMonthAccTable").style.display = "none";
         document.getElementById("oneYearAccTable").style.display = "none";
     }else if(selected == "sixMonthAcc") {

         document.getElementById("maxAccTable").style.display = "none";
         document.getElementById("oneWeekAccTable").style.display = "none";
         document.getElementById("oneMonthAccTable").style.display = "none";
         document.getElementById("sixMonthAccTable").style.display = "block";
         document.getElementById("oneYearAccTable").style.display = "none";
     }else if(selected == "oneYearAcc") {

         document.getElementById("maxAccTable").style.display = "none";
         document.getElementById("oneWeekAccTable").style.display = "none";
         document.getElementById("oneMonthAccTable").style.display = "none";
         document.getElementById("sixMonthAccTable").style.display = "none";
         document.getElementById("oneYearAccTable").style.display = "block";
     }

}