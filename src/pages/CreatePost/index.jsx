import "./style.less";

const CreatePost = () => {
  return (
    <div className="CreatePost">
		<header className="CreatePost-header">
			<img src={logo} alt="logo"/>
			<p>Logout</p>
		</header>
      	<div className="App-body">
            <p>username</p>
            <input type="text" placeholder="Put Title Here" />
            <input type="text" placeholder="Body of Content" />

            <button>Submit</button>
      	</div>
    </div>
  );
}

export default CreatePost;