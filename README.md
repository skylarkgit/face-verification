# face-verification
Verify if the input face matches the face in store

## Download the model from here
https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

## CURL Request
curl -F 'target=@/path/to/target.jpg' -F 'source=@/path/to/source.jpg' http://127.0.0.1:5002/verify