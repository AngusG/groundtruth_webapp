"use strict";module.exports=function(r){if(!Array.isArray(r))throw new TypeError("array-unique expects an array.");for(var e=r.length,a=-1;a++<e;)for(var t=a+1;t<r.length;++t)r[a]===r[t]&&r.splice(t--,1);return r};