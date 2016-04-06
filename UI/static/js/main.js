// console.log("js working");


// if (FileReader) {
// 	function display(event) {
// 		console.log("function working");
// 		var f = document.getElementById("fileinput");
// 		// var f = event.target.files[0];
// 		if (f) {
// 			// console.log(f);
// 			var file = f.files[0];
// 			var fr = new FileReader();
// 			fr.onload = function() {
// 				console.log(fr.result);
// 				document.getElementById("file-content").innerHTML = fr.result;
// 				var req = new XMLHttpRequest();
// 				// req.onreadystatechange = getResponse;
// 				req.open("POST", '/', true);
// 				req.setRequestHeader(
// 					"Content-type",
// 					"application/json"
// 				);
// 				req.send(JSON.stringify(fr.result));
// 			};
// 			fr.readAsText(file);
// 		}
// 		else { 
//       		console.log("Failed to load file");
//     	}
// 	}
// }
