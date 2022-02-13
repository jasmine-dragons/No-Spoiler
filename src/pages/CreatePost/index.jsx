import "./style.less";
import logo from '../../assets/logo.svg';

const CreatePost = () => {
  	return (
		<div className="CreatePost">
			<header className="CreatePost-header">
				<img src={logo} alt="logo"/>
				<p>Logout</p>
			</header>
			<div className="CreatePost-body">
				<div className="post">
					<p>username</p>
					<input className="title" type="text" placeholder="Put Title Here" />
					<div className="blank"></div>
					<textarea className="content" placeholder="Body of Content" />
				</div>

				<div className="buttons">
					<button>Cancel</button>
					<button>Submit</button>
				</div>
			</div>
		</div>
  	);
}

export default CreatePost;