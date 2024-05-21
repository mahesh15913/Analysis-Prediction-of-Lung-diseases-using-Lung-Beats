function aboutus(){
    window.location.href = "aboutus.html"
}
function Home(){
    window.location.href = "index.html"
}
function analysis(){
    window.location.href = "analysis.html"
}
function prediction(){
    window.location.href = "prediction.html"
}
function project(){
    window.location.href = "project.html"
}
function project_code(){
    window.location.href = 'https://github.com/mahesh15913/Analysis-Prediction-of-Lung-diseases-using-Lung-Beats'
}
function email(){
    window.location.href = 'https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox?compose=GTvVlcSBpgRQXSjVJSpMJGNrXKpSwHPcbpkRMhRTVklzNqCgnPnQwnXpNqtkzdcXPrXTSDvXZxvzg'
}
function literature(){
    window.location.href = 'https://drive.google.com/file/d/1XKUObNXhzhPm3Y1MX3stm05Rbuek5zSv/view?usp=sharing'
}
const imageGallery = document.querySelector('.image-gallery');
imageGallery.addEventListener('wheel', (event) => {
    event.preventDefault();
    imageGallery.scrollLeft += event.deltaY;
});