css='''
<style>
.chat-message{
    padding:1rem; border-radius:0.5rem; margin-bottom:1rem; display:flex;
}
.chat-message.user{
    background-color:#2b313e;
}
.chat-message.bot{
    background-color:#475063;
}
.chat-message .avatar{
    width:7%;
}
.chat-message .avatar img{
    max-width:28px;
    max-height:28px;
    border-radius:50%;
    object-fit:cover;
}
.chat-message .message{
    width:90%;
    padding:0 1rem;
    color:#fff;
}
'''
bot_template='''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,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" >
    </div>
    <div class="message">{{MSG}}</div>
''' 
user_template='''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7OJcGATX7ZH17Fi0l7Yeb-6nzfjWP9c4Z9A&usqp=CAU" >
    </div>
    <div class="message">{{MSG}}</div>
''' 
