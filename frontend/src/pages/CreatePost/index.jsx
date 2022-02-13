import "./style.less";
import logo from '../../assets/logo.svg';
import { useState } from "react";

const CreatePost = () => {
    const [title, setTitle] = useState("");
    const [content, setContent] = useState("");

    const submitPost = () => {
        fetch('http://da08-2603-8000-8e00-33fb-81a4-359e-8c3f-1694.ngrok.io/create', {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: localStorage.getItem("username"),
                title: title,
                content: content
            })
        })
        .then(response => response.json())
        .then(data => console.log(data));

        window.location.href = "/";
    }
  	return (
		<div className="CreatePost">
			<header className="CreatePost-header">
				<img src={logo} alt="logo"/>
				<p onClick={() => {window.location.href = "/signin"}}>Logout</p>
			</header>
			<div className="CreatePost-body">
				<div className="post">
					<p>{localStorage.getItem("username")}</p>
					<input className="title" type="text" placeholder="Put Title Here" value={title} onChange={e => setTitle(e.target.value)}/>
					<div className="blank"></div>
					<textarea className="content" placeholder="Body of Content" value={content} onChange={e => setContent(e.target.value)}/>
				</div>

				<div className="buttons">
					<button onClick={() => window.location.href = "/"}>Cancel</button>
					<button onClick={submitPost}>Submit</button>
				</div>
			</div>
		</div>
  	);
}

export default CreatePost;