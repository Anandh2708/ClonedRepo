
function Sgraph(){
   
    var e = document.getElementById("predictionType");
    var selected = e.options[e.selectedIndex].value;

    if (selected == "regularLSTM"){

         document.getElementById("regular").style.display = "block";
         document.getElementById("crude").style.display = "none";
         document.getElementById("interest").style.display = "none";
         document.getElementById("hybrid").style.display = "none";
         document.getElementById("weighted").style.display = "none";
    }else if(selected == "crudeLSTM") {
        //document.getElementById("try").innerHTML = "reg";
         document.getElementById("regular").style.display = "none";
         document.getElementById("crude").style.display = "block";
         document.getElementById("interest").style.display = "none";
         document.getElementById("hybrid").style.display = "none";
         document.getElementById("weighted").style.display = "none";
    }else if(selected == "interestLSTM"){
         document.getElementById("regular").style.display = "none";
         document.getElementById("crude").style.display = "none";
         document.getElementById("interest").style.display = "block";
         document.getElementById("hybrid").style.display = "none";
         document.getElementById("weighted").style.display = "none";
    }else if(selected == "hybridLSTM"){
         document.getElementById("regular").style.display = "none";
         document.getElementById("crude").style.display = "none";
         document.getElementById("interest").style.display = "none";
         document.getElementById("hybrid").style.display = "block";
         document.getElementById("weighted").style.display = "none";
    }
    else if(selected == "weightedLSTM"){

         document.getElementById("regular").style.display = "none";
         document.getElementById("crude").style.display = "none";
         document.getElementById("interest").style.display = "none";
         document.getElementById("hybrid").style.display = "none";
         document.getElementById("weighted").style.display = "block";
    }


}

