import "./style.less";
import logo from '../../assets/logo.svg';
import { useState } from "react";

const CreatePost = () => {
    const [username, setUsername] = useState("");
    const [title, setTitle] = useState("");
    const [content, setContent] = useState("");

    const submitPost = () => {
        fetch('https://localhost/5000/create', {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: username,
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
				<p>Logout</p>
			</header>
			<div className="CreatePost-body">
				<div className="post">
					<p>{username}</p>
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