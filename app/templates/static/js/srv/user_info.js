angular.module('myApp').service('user_info', function() {

    this.user_info_object = {
          "data": {
            "type": "users",
            "attributes": {
              "email":"sample@email",
              "full_name": "Dr. Smith",
              "classified": "0",
              "in_queue": "0",
              "percent_complete": "0",
              "image_info": {}
              }
           }
        };

    this.setImageData = function(image_info){
      this.user_info_object.data.attributes.image_info = image_info;
      //alert(this.user_info_object.data.attributes.image_info[0].fullsize_orig_filepath);
    };

    this.setFullName = function(full_name){
      this.user_info_object.data.attributes.full_name = full_name;
    };
    this.sayHello = function(){
      return "Hello " + user_info_object.data.attributes.full_name;
    };

    this.setClassified = function(classified){
      this.user_info_object.data.attributes.classified = classified;
    };
    this.getClassified = function(){
      return user_info_object.data.attributes.classified;
    };

    this.setInQueue = function(in_queue){
      this.user_info_object.data.attributes.in_queue = in_queue;
    };
    this.getInQueue = function(){
      return user_info_object.data.attributes.in_queue;
    };

    this.setPercentComplete = function(pComplete){
      this.user_info_object.data.attributes.percent_complete = pComplete;
    };
    this.getPercentComplete = function(){
      return this.user_info_object.data.attributes.percent_complete;
    };


});
