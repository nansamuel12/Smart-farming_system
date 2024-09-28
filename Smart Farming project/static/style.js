const popup = document.querySelector('.chat_pop_up');
const chatBtn = document.querySelector('.chat_btn');

//  for pop up
chatBtn.addEventListener('click', () => {
    popup.classList.toggle('show');
})


const pop = document.querySelector('.chat_pop_up');
const anchor_btn = document.querySelector('.header_btn');
anchor_btn.addEventListener('click', () => {
    pop.classList.toggle('show');
})


const pop1= document.querySelector('.chat_pop_up');
const anchor_btn1= document.querySelector('.pop_chat_btn');
anchor_btn1.addEventListener('click', () => {
    pop1.classList.toggle('show');
})