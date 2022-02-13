import "./style.less";
import logo from '../../assets/logo.svg';
import { useState } from "react";

const CreatePost = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");

    const signIn = () => {
        fetch('http://da08-2603-8000-8e00-33fb-81a4-359e-8c3f-1694.ngrok.io/signin', {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: username,
                password: password,
            })
        })
        .then(response => response.json())
        .then(data => console.log(data));

        localStorage.setItem("username", username);

        window.location.href = "/";
    }
  	return (
		<div className="SignIn">
			<header className="SignIn-header">
				<img src={logo} alt="logo"/>
			</header>
			<div className="SignIn-body">
				<div className="signin">
					<p>Sign In</p>
					<input className="username" type="text" placeholder="username" value={username} onChange={e => setUsername(e.target.value)}/>
					<div className="blank"></div>
					<input className="password" type="password" placeholder="password" value={password} onChange={e => setPassword(e.target.value)} />
				</div>

				<div className="buttons">
					<button onClick={signIn}>Sign In</button>
				</div>
			</div>
		</div>
  	);
}

export default CreatePost;